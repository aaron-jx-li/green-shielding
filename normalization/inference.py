#!/usr/bin/env python3
import json
import argparse
import time
from typing import Optional, Any, List, Dict

from tqdm import tqdm
from openai import OpenAI

DEFAULT_INSTRUCTION = (
    "You are a careful medical assistant helping a real patient. "
    "Based on the information provided, explain what the most likely diagnoses are, "
    "focusing on a small number of clinically plausible explanations. "
    "When appropriate, briefly mention any serious or time-sensitive diagnoses "
    "that a clinician would want to actively consider or rule out. "
    "If the information is insufficient to be confident, explicitly acknowledge this. "
    "Be concise, clinically realistic, and avoid inventing details."
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Query an OpenAI model for JSON samples using raw + normalized inputs."
    )

    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to input JSON file (list of samples).")

    parser.add_argument("--output_path", type=str, required=True,
                        help="Where to save the output JSON (new file) with model responses added.")

    parser.add_argument("--model", type=str, default="gpt-4.1-mini",
                        help="Model name, e.g. gpt-4o, gpt-4o-mini, gpt-5.1, etc.")

    # Keys to read from each sample
    parser.add_argument("--raw_key", type=str, default="raw_input",
                        help='JSON key for the raw patient text (default: "raw_input").')

    parser.add_argument("--normalized_key", type=str, default="normalized_prompt",
                        help='JSON key for the normalized prompt (default: "normalized_prompt").')

    # Keys to write outputs into
    parser.add_argument("--out_raw_key", type=str, default="model_response_raw",
                        help='JSON key to store model output for raw input (default: "model_response_raw").')

    parser.add_argument("--out_normalized_key", type=str, default="model_response_converted",
                        help='JSON key to store model output for normalized prompt (default: "model_response_converted").')

    # Prompting controls
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION,
                        help="Developer instruction to send to the model.")

    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (ignored for gpt-5* models).")

    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Max output tokens (ignored for gpt-5* models).")

    # Index slicing
    parser.add_argument("--start_idx", type=int, default=None,
                        help="Start index (inclusive).")

    parser.add_argument("--end_idx", type=int, default=None,
                        help="End index (inclusive).")

    # Optional rate limiting
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Seconds to sleep between requests (default: 0.0).")

    # If set, skip a query if the output key already exists and is non-empty
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip querying a field if its output key already exists and is non-empty (within processed indices).")

    return parser.parse_args()


def query_model(
    client: OpenAI,
    input_text: str,
    model: str,
    instruction: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> str:
    messages = [
        {"role": "developer", "content": instruction},
        {"role": "user", "content": input_text},
    ]

    if "gpt-5" in model:
        resp = client.responses.create(model=model, input=messages)
    else:
        resp = client.responses.create(
            model=model,
            input=messages,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    return resp.output_text or ""


def _nonempty(x: Any) -> bool:
    return isinstance(x, str) and x.strip() != ""


def main():
    args = parse_args()
    client = OpenAI()  # requires OPENAI_API_KEY

    # --------------------------
    # Load data
    # --------------------------
    with open(args.input_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("❌ Input JSON must be a list of samples (each a dict).")

    total = len(data)
    print(f"Loaded {total} samples from {args.input_path}.")

    # Determine processing range
    if args.start_idx is None and args.end_idx is None:
        start_idx, end_idx = 0, total - 1
        print("No start/end index specified — processing ALL samples.")
    else:
        if args.start_idx is None or args.end_idx is None:
            raise ValueError("❌ You must specify BOTH --start_idx and --end_idx.")

        start_idx = max(0, args.start_idx)
        end_idx = min(total - 1, args.end_idx)

        if start_idx > end_idx:
            raise ValueError(f"❌ Invalid range: start_idx={args.start_idx} > end_idx={args.end_idx}")

        print(f"Processing samples from index {start_idx} to {end_idx} inclusive.")

    processed_set = set(range(start_idx, end_idx + 1))

    # --------------------------
    # First pass: remove model_response_* from UNPROCESSED items
    # --------------------------
    for i, sample in enumerate(data):
        if not isinstance(sample, dict):
            continue
        if i not in processed_set:
            sample.pop(args.out_raw_key, None)
            sample.pop(args.out_normalized_key, None)

    # --------------------------
    # Second pass: process selected samples
    # --------------------------
    for idx in tqdm(sorted(processed_set)):
        if idx < 0 or idx >= total:
            print(f"❌ Index {idx} is out of bounds. Skipping.")
            continue

        sample = data[idx]
        if not isinstance(sample, dict):
            print(f"❌ Sample {idx} is not a JSON object/dict. Skipping.")
            continue

        raw_text = sample.get(args.raw_key, "")
        norm_text = sample.get(args.normalized_key, "")

        # Query on raw input
        if raw_text:
            if args.skip_existing and _nonempty(sample.get(args.out_raw_key, "")):
                pass
            else:
                try:
                    out_raw = query_model(
                        client,
                        raw_text,
                        model=args.model,
                        instruction=args.instruction,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )
                    if not out_raw.strip():
                        raise ValueError("Empty model response")
                    sample[args.out_raw_key] = out_raw
                except Exception as e:
                    print(f"❌ Error querying RAW for sample {idx}: {e}")
                    sample[args.out_raw_key] = None

                if args.sleep > 0:
                    time.sleep(args.sleep)

        # Query on normalized prompt
        if norm_text:
            if args.skip_existing and _nonempty(sample.get(args.out_normalized_key, "")):
                pass
            else:
                try:
                    out_norm = query_model(
                        client,
                        norm_text,
                        model=args.model,
                        instruction=args.instruction,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )
                    if not out_norm.strip():
                        raise ValueError("Empty model response")
                    sample[args.out_normalized_key] = out_norm
                except Exception as e:
                    print(f"❌ Error querying NORMALIZED for sample {idx}: {e}")
                    sample[args.out_normalized_key] = None

                if args.sleep > 0:
                    time.sleep(args.sleep)

    # --------------------------
    # Save output JSON (new file)
    # --------------------------
    with open(args.output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✨ Saved updated dataset to {args.output_path}")


if __name__ == "__main__":
    main()
