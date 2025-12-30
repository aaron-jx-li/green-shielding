import json
import argparse
import time
from openai import OpenAI
from tqdm import tqdm
from typing import Dict, List, Optional

DEFAULT_INSTRUCTION = "You are a helpful medical assistant. Given the description of the symptoms, what is the most likely diagnosis? Be concise and accurate."

def parse_args():
    parser = argparse.ArgumentParser(description="Query OpenAI model for JSON samples.")

    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to input JSON file (list of samples).")

    parser.add_argument("--output_path", type=str, required=True,
                        help="Where to save the output JSON with model responses added.")

    parser.add_argument("--model", type=str, default="gpt-4.1-mini",
                        help="Model name, e.g. gpt-4o, gpt-4o-mini, gpt-5.1, etc.")
    
    parser.add_argument("--col_name", type=str, required=True,
                    help="Column name for target response (e.g., 'model_response').")

    parser.add_argument("--start_idx", type=int, default=None,
                        help="Start index (inclusive).")

    parser.add_argument("--end_idx", type=int, default=None,
                        help="End index (inclusive).")

    return parser.parse_args()


def query_model(client, input_text, model="gpt-4.1-mini", instruction=DEFAULT_INSTRUCTION, temperature: float = 0.0, max_tokens: Optional[int] = None):
    """
    Sends prompt to OpenAI model and returns response string.
    """
    
    messages = [
        {"role": "developer", "content": instruction},
        {"role": "user", "content": input_text},
    ]

    if "gpt-5" in model: # newer models don't support temperature or max_output_tokens
        resp = client.responses.create(
            model=model,
            input=messages,
        )
    else:
        resp = client.responses.create(
            model=model,
            input=messages,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    return resp.output_text or ""


def main():
    args = parse_args()
    client = OpenAI()  # needs OPENAI_API_KEY env var

    # --------------------------
    # Load data
    # --------------------------
    with open(args.input_path, "r") as f:
        data = json.load(f)
    total = len(data)
    print(f"Loaded {len(data)} samples from {args.input_path}.")

    if args.start_idx is None and args.end_idx is None:
        # default: all samples
        start_idx = 0
        end_idx = total - 1
        print("No start/end index specified — processing ALL samples.")
    else:
        if args.start_idx is None or args.end_idx is None:
            raise ValueError("❌ You must specify BOTH --start_idx and --end_idx.")

        start_idx = max(0, args.start_idx)
        end_idx = min(total - 1, args.end_idx)

        if start_idx > end_idx:
            raise ValueError(f"❌ Invalid range: start_idx={args.start_idx} > end_idx={args.end_idx}")

        print(f"Processing samples from index {start_idx} to {end_idx} inclusive.")

    # Final list of indices
    indices = list(range(start_idx, end_idx + 1))
    # --------------------------
    # Process selected samples
    # --------------------------
    for idx in tqdm(indices):
        if idx < 0 or idx >= len(data):
            print(f"❌ Index {idx} is out of bounds. Skipping.")
            continue

        sample = data[idx]
        # print(f"Processing sample {idx}...")

        # instruction = sample.get("instruction", "")
        # input_text = sample.get("normalized_prompt", "")
        input_text = sample.get(args.col_name, "")

        try:
            model_resp = query_model(client, input_text, args.model)
            # Treat empty output as an error
            if not model_resp or model_resp.strip() == "":
                raise ValueError("Empty model response")
        except Exception as e:
            print(f"❌ Error querying sample {idx}: {e}")
            model_resp = None

        sample["model_response"] = model_resp

        # time.sleep(0.2)  # optional polite rate-limit

    # --------------------------
    # Save output JSON
    # --------------------------
    with open(args.output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✨ Saved updated dataset with model responses to {args.output_path}")


if __name__ == "__main__":
    main()
