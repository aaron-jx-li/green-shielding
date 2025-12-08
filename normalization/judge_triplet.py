import json
import argparse
import re
from collections import Counter
from tqdm import tqdm
from openai import OpenAI
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ranking judge for three outputs: Original Output, Original Model Output, and Converted Prompt Model Output."
    )

    # Input file (single file containing all four components)
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="CSV file with columns: original_user_input, original_output, original_model_output, converted_model_output.",
    )

    # Output
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Where to save judge results.",
    )

    # Judge model
    parser.add_argument(
        "--judge_model",
        type=str,
        default="gpt-4.1-mini",
        help="Model used for judging.",
    )

    # Field names (user configurable)
    parser.add_argument(
        "--original_user_input_field",
        type=str,
        default="input",
        help="Field name for original user input in input_path.",
    )
    parser.add_argument(
        "--original_output_field",
        type=str,
        default="output",
        help="Field name for original output (reference) in input_path.",
    )
    parser.add_argument(
        "--original_model_output_field",
        type=str,
        default="model_response",
        help="Field name for original model output in input_path.",
    )
    parser.add_argument(
        "--converted_model_output_field",
        type=str,
        default="converted_model_response",
        help="Field name for converted prompt model output in input_path.",
    )

    return parser.parse_args()


def build_ranking_judge_messages(
    original_user_input: str,
    original_output: str,
    original_model_output: str,
    converted_model_output: str,
):
    """
    Build messages for ranking three outputs:
    1. Original Output (reference from doctors)
    2. Original Model Output (model's response to original prompt)
    3. Converted Prompt Model Output (model's response to converted prompt)
    
    The judge ranks these three outputs based on which provides the best likely diagnosis
    for the patient asking the original question.
    """
    msgs = [
        {
            "role": "developer",
            "content": (
                "You are a strict medical evaluation assistant.\n\n"
                "You will be given:\n"
                "- ORIGINAL_USER_INPUT (the original question/input from the patient)\n"
                "- ORIGINAL_OUTPUT (the reference response from experienced clinicians)\n"
                "- ORIGINAL_MODEL_OUTPUT (the model's response to the original prompt)\n"
                "- CONVERTED_PROMPT_MODEL_OUTPUT (the model's response to a converted/normalized prompt)\n\n"
                "Your task is to rank the THREE OUTPUTS (Original Output, Original Model Output, "
                "and Converted Prompt Model Output) from BEST to WORST based on which provides "
                "the best likely diagnosis for the patient asking the original question.\n\n"
                "VERY IMPORTANT RULES:\n"
                "- Focus **exclusively** on diagnostic correctness and diagnostic specificity.\n"
                "- Rank based on how well each output identifies the most likely diagnosis for the patient.\n"
                "- Ignore style, tone, empathy, length, clarity, bedside manner, disclaimers, "
                "safety language, or any other non-diagnostic elements.\n"
                "- Ignore supportive reasoning, pathophysiology explanations, and narrative details "
                "unless they change the explicit or implied diagnosis.\n"
                "- Do NOT reward verbosity or penalize brief answers.\n"
                "- Do NOT consider treatment recommendations unless they directly indicate the diagnosis.\n"
                "- The ranking should reflect: Which output gives the most accurate and specific "
                "diagnosis for this patient's case?\n\n"
                "Output format (IMPORTANT):\n"
                "Line 1: RANKING in the format: \"1st: [OUTPUT_NAME], 2nd: [OUTPUT_NAME], 3rd: [OUTPUT_NAME]\"\n"
                "        Where OUTPUT_NAME is one of: ORIGINAL_OUTPUT, ORIGINAL_MODEL_OUTPUT, or CONVERTED_PROMPT_MODEL_OUTPUT\n"
                "Line 2: A brief justification (<200 words) explaining the ranking, focusing ONLY on diagnostic correctness."
            ),
        },
        {
            "role": "user",
            "content": (
                f"ORIGINAL_USER_INPUT:\n{(original_user_input or '').strip()}\n\n"
                f"ORIGINAL_OUTPUT:\n{(original_output or '').strip()}\n\n"
                f"ORIGINAL_MODEL_OUTPUT:\n{(original_model_output or '').strip()}\n\n"
                f"CONVERTED_PROMPT_MODEL_OUTPUT:\n{(converted_model_output or '').strip()}\n\n"
                "Rank the three outputs (ORIGINAL_OUTPUT, ORIGINAL_MODEL_OUTPUT, CONVERTED_PROMPT_MODEL_OUTPUT) "
                "from best to worst based on which provides the best likely diagnosis for the patient.\n\n"
                "Respond in exactly two parts:\n"
                "Line 1: RANKING in format: \"1st: [OUTPUT_NAME], 2nd: [OUTPUT_NAME], 3rd: [OUTPUT_NAME]\"\n"
                "Line 2: A brief justification (<200 words) explaining the ranking."
            ),
        },
    ]
    return msgs


def normalize_ranking(raw_ranking: str) -> dict:
    """
    Parse ranking from LLM response and return a dictionary with:
    {
        "1st": "OUTPUT_NAME",
        "2nd": "OUTPUT_NAME", 
        "3rd": "OUTPUT_NAME"
    }
    
    Valid OUTPUT_NAME values: ORIGINAL_OUTPUT, ORIGINAL_MODEL_OUTPUT, CONVERTED_PROMPT_MODEL_OUTPUT
    
    Returns None if parsing fails.
    """
    if not raw_ranking:
        return None

    text = raw_ranking.strip().upper()
    
    # Try to extract ranking from format like "1ST: ORIGINAL_OUTPUT, 2ND: ORIGINAL_MODEL_OUTPUT, 3RD: CONVERTED_PROMPT_MODEL_OUTPUT"
    ranking = {}
    
    # Look for patterns like "1ST:", "2ND:", "3RD:" or "1:", "2:", "3:"
    
    # Pattern to match "1st: OUTPUT_NAME" or "1: OUTPUT_NAME"
    pattern = r'(?:1ST|1):\s*([A-Z_]+)'
    match = re.search(pattern, text)
    if match:
        ranking["1st"] = match.group(1)
    
    pattern = r'(?:2ND|2):\s*([A-Z_]+)'
    match = re.search(pattern, text)
    if match:
        ranking["2nd"] = match.group(1)
    
    pattern = r'(?:3RD|3):\s*([A-Z_]+)'
    match = re.search(pattern, text)
    if match:
        ranking["3rd"] = match.group(1)
    
    # Normalize output names
    valid_names = {
        "ORIGINAL_OUTPUT": "ORIGINAL_OUTPUT",
        "ORIGINAL_MODEL_OUTPUT": "ORIGINAL_MODEL_OUTPUT",
        "CONVERTED_PROMPT_MODEL_OUTPUT": "CONVERTED_PROMPT_MODEL_OUTPUT",
        "ORIGINAL": "ORIGINAL_OUTPUT",
        "ORIGINAL_MODEL": "ORIGINAL_MODEL_OUTPUT",
        "CONVERTED": "CONVERTED_PROMPT_MODEL_OUTPUT",
        "CONVERTED_PROMPT": "CONVERTED_PROMPT_MODEL_OUTPUT",
    }
    
    for key in ["1st", "2nd", "3rd"]:
        if key in ranking:
            original_name = ranking[key]
            ranking[key] = valid_names.get(original_name, original_name)
    
    # Check if we got all three rankings
    if len(ranking) == 3 and all(k in ranking for k in ["1st", "2nd", "3rd"]):
        # Verify all are valid
        if all(v in valid_names.values() for v in ranking.values()):
            return ranking
    
    return None


def call_judge(client, judge_model: str, msgs):
    """
    Call the judge and return (ranking_dict, justification).
    ranking_dict is a dict with keys "1st", "2nd", "3rd" mapping to output names.
    Returns (None, "") if parsing fails.
    """
    if "gpt-5" in judge_model:
        resp = client.responses.create(
            model=judge_model,
            input=msgs,
        )
    else:
        resp = client.responses.create(
            model=judge_model,
            input=msgs,
            temperature=0.0,
        )

    raw = (resp.output_text or "").strip()
    if not raw:
        return None, ""

    lines = raw.splitlines()
    raw_ranking = lines[0] if lines else ""
    justification = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
    
    # Limit justification to 200 words
    if justification:
        words = justification.split()
        if len(words) > 200:
            justification = " ".join(words[:200]) + "..."

    ranking = normalize_ranking(raw_ranking)
    return ranking, justification


def get_field(sample: dict, field_name: str):
    """Safe getter: returns None if sample is not a dict or field_name is None."""
    if not isinstance(sample, dict) or field_name is None:
        return None
    return sample.get(field_name)


def main():
    args = parse_args()
    client = OpenAI()


    all_data_df = pd.read_csv(args.input_path)

    print(f"Loaded {len(all_data_df)} samples.")

    results = []
    num_total = 0
    num_failed = 0
    
    # Track ranking statistics
    first_place_counts = Counter()
    second_place_counts = Counter()
    third_place_counts = Counter()

    for idx, sample in tqdm(all_data_df.iterrows(), desc="Judging", total=len(all_data_df)):
        # Extract all four components
        original_user_input = sample[args.original_user_input_field]
        original_output = sample[args.original_output_field]
        original_model_output = sample[args.original_model_output_field]
        converted_model_output = sample[args.converted_model_output_field]

        # If any critical parts missing, skip this sample
        if not all([original_user_input, original_output, original_model_output, converted_model_output]):
            continue

        msgs = build_ranking_judge_messages(
            original_user_input=original_user_input,
            original_output=original_output,
            original_model_output=original_model_output,
            converted_model_output=converted_model_output,
        )

        ranking, justification = call_judge(client, args.judge_model, msgs)
        
        if ranking is None:
            num_failed += 1
            continue
        
        num_total += 1
        
        # Track statistics
        first_place_counts[ranking["1st"]] += 1
        second_place_counts[ranking["2nd"]] += 1
        third_place_counts[ranking["3rd"]] += 1

        results.append(
            {
                "index": idx,
                "ranking": ranking,  # {"1st": "...", "2nd": "...", "3rd": "..."}
                "justification": justification,
                "original_user_input": original_user_input,
                "original_output": original_output,
                "original_model_output": original_model_output,
                "converted_model_output": converted_model_output,
            }
        )

        # Intermediate save every 10 judged questions
        if num_total % 10 == 0:
            summary_intermediate = {
                "num_total_judged": num_total,
                "num_failed_parsing": num_failed,
                "first_place_counts": dict(first_place_counts),
                "second_place_counts": dict(second_place_counts),
                "third_place_counts": dict(third_place_counts),
                "judge_model": args.judge_model,
                "config": {
                    "original_user_input_field": args.original_user_input_field,
                    "original_output_field": args.original_output_field,
                    "original_model_output_field": args.original_model_output_field,
                    "converted_model_output_field": args.converted_model_output_field,
                },
            }

            output_intermediate = {
                "summary": summary_intermediate,
                "per_sample": results,
            }

            intermediate_path = args.output_path.replace(".json", f"_intermediate_{num_total}.json")
            with open(intermediate_path, "w") as f_inter:
                json.dump(output_intermediate, f_inter, indent=2, ensure_ascii=False)
            print(f"Intermediate save at {num_total} judged questions: {intermediate_path}")

    # Build summary
    summary = {
        "num_total_judged": num_total,
        "num_failed_parsing": num_failed,
        "first_place_counts": dict(first_place_counts),
        "second_place_counts": dict(second_place_counts),
        "third_place_counts": dict(third_place_counts),
        "judge_model": args.judge_model,
        "config": {
            "original_user_input_field": args.original_user_input_field,
            "original_output_field": args.original_output_field,
            "original_model_output_field": args.original_model_output_field,
            "converted_model_output_field": args.converted_model_output_field,
        },
    }

    output = {
        "summary": summary,
        "per_sample": results,
    }

    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("=== JUDGE SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved results to {args.output_path}")


if __name__ == "__main__":
    main()