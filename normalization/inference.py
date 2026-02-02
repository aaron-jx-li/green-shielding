import json
import argparse
import time
import os
import tempfile
from typing import Optional, Any

from tqdm import tqdm
from openai import OpenAI

DEFAULT_INSTRUCTION = (
   "You are a helpful medical assistant. "
   "Please answer the medical question given by the user. "
   "Be concise, accurate, and clinically realistic."
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

    # Base keys to write outputs into (script will append _1.._n)
    parser.add_argument("--out_raw_key", type=str, default="model_response_raw",
                        help='Base JSON key to store model output for raw input (default: "model_response_raw").')

    parser.add_argument("--out_normalized_key", type=str, default="model_response_converted",
                        help='Base JSON key to store model output for normalized prompt (default: "model_response_converted").')

    # Multi-run controls
    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of independent runs per input (default: 1). Keys will be suffixed _1.._n.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional seed for reproducibility (if supported by the model).")

    # Prompting controls
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION,
                        help="Developer instruction to send to the model.")

    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (ignored for some models).")

    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Max output tokens.")

    # Index slicing
    parser.add_argument("--start_idx", type=int, default=None,
                        help="Start index (inclusive).")

    parser.add_argument("--end_idx", type=int, default=None,
                        help="End index (inclusive).")

    # Optional rate limiting
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Seconds to sleep between requests (default: 0.0).")

    # If set, skip a query if *all* run keys already exist and are non-empty
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip querying if all output keys for all runs already exist and are non-empty.")

    # Periodic checkpoint saving
    parser.add_argument("--save_every", type=int, default=50,
                        help="Save a checkpoint every N processed samples (default: 50). Set <=0 to disable.")

    return parser.parse_args()


def atomic_json_dump(obj: Any, path: str) -> None:
    """
    Atomic-ish save: write to a temp file on the same filesystem then replace.
    Prevents corrupt JSON if the job is interrupted mid-write.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=os.path.dirname(path) or ".")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def query_model(
    client: OpenAI,
    input_text: str,
    model: str,
    instruction: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    num_runs: int = 1,
    seed: Optional[int] = None,
) -> list[str]:
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input_text},
    ]

    outs: list[str] = []
    for k in range(num_runs):
        # Base kwargs for chat completions
        kwargs = {
            "model": model,
            "messages": messages,  # standard API uses 'messages', not 'input'
        }

        # Handling model-specific parameter constraints
        if model.startswith("o1") or model.startswith("deepseek-reasoner"):
            # Reasoning models often don't support temperature (fixed at 1)
            pass
        elif temperature is not None:
            kwargs["temperature"] = temperature

        if max_tokens is not None:
            # Standard OpenAI API uses max_tokens (or max_completion_tokens for o1)
            if model.startswith("o1") or model.startswith("deepseek-reasoner"):
                 kwargs["max_completion_tokens"] = max_tokens
            else:
                 kwargs["max_tokens"] = max_tokens
        
        if seed is not None:
            kwargs["seed"] = seed + k

        try:
            # Use standard chat completion
            resp = client.chat.completions.create(**kwargs)
            outs.append(resp.choices[0].message.content or "")
        except Exception as e:
            # Fallback for custom/legacy client if standard fails
            print(f"Standard chat completion failed: {e}. Trying legacy/custom...")
            try:
                # Legacy custom wrapper adapt
                kwargs["input"] = kwargs.pop("messages") 
                if "max_tokens" in kwargs:
                    kwargs["max_output_tokens"] = kwargs.pop("max_tokens")
                
                resp = client.responses.create(**kwargs)
                outs.append(getattr(resp, "output_text", "") or "")
            except Exception as e2:
                print(f"Custom client also failed: {e2}")
                raise e

    return outs


def _nonempty(x: Any) -> bool:
    return isinstance(x, str) and x.strip() != ""


def _gen_key(base: str, k: int) -> str:
    return f"{base}_{k}"


def _all_runs_present(sample: dict, base_key: str, n: int) -> bool:
    return all(_nonempty(sample.get(_gen_key(base_key, k), "")) for k in range(1, n + 1))


def _pop_run_keys(sample: dict, base_key: str, n: int) -> None:
    for k in range(1, n + 1):
        sample.pop(_gen_key(base_key, k), None)


def main():
    args = parse_args()
    if args.num_runs < 1:
        raise ValueError("❌ --num_runs must be >= 1")

    # If using DeepSeek, we need to configure the client appropriately
    # The shell script exports DEEPSEEK_API_KEY, but OpenAI() client looks for OPENAI_API_KEY.
    # Also need to set base_url for DeepSeek.
    
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    # Check if DEEPSEEK_API_KEY is present and we are using a DeepSeek model
    if "deepseek" in args.model.lower():
        ds_key = os.getenv("DEEPSEEK_API_KEY")
        if ds_key:
            api_key = ds_key
            base_url = "https://api.deepseek.com"
            print(f"🔹 Detected DeepSeek model '{args.model}' and DEEPSEEK_API_KEY. Configured client for DeepSeek.")
    
    client = OpenAI(api_key=api_key, base_url=base_url)

    # print("=== DEBUG ===")
    # print("model:", args.model)
    # print("DEEPSEEK_API_KEY set?:", bool(os.getenv("DEEPSEEK_API_KEY")))
    # print("OPENAI_API_KEY set?:", bool(os.getenv("OPENAI_API_KEY")))
    # print("base_url:", base_url)
    # print("api_key prefix:", (api_key or "")[:8])
    # print("=============")


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
    # First pass: remove run keys from UNPROCESSED items
    # --------------------------
    for i, sample in enumerate(data):
        if not isinstance(sample, dict):
            continue
        if i not in processed_set:
            _pop_run_keys(sample, args.out_raw_key, args.num_runs)
            _pop_run_keys(sample, args.out_normalized_key, args.num_runs)

    # --------------------------
    # Second pass: process selected samples
    # --------------------------
    processed_count = 0
    last_save_time = time.time()

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
            if not (args.skip_existing and _all_runs_present(sample, args.out_raw_key, args.num_runs)):
                try:
                    outs_raw = query_model(
                        client,
                        raw_text,
                        model=args.model,
                        instruction=args.instruction,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        num_runs=args.num_runs,
                        seed=args.seed,
                    )
                    for k, out_raw in enumerate(outs_raw, start=1):
                        sample[_gen_key(args.out_raw_key, k)] = out_raw if out_raw.strip() else None
                except Exception as e:
                    print(f"❌ Error querying RAW for sample {idx}: {e}")
                    for k in range(1, args.num_runs + 1):
                        sample[_gen_key(args.out_raw_key, k)] = None

                if args.sleep > 0:
                    time.sleep(args.sleep)

        # Query on normalized prompt
        if norm_text:
            if not (args.skip_existing and _all_runs_present(sample, args.out_normalized_key, args.num_runs)):
                try:
                    outs_norm = query_model(
                        client,
                        norm_text,
                        model=args.model,
                        instruction=args.instruction,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        num_runs=args.num_runs,
                        seed=args.seed,
                    )
                    for k, out_norm in enumerate(outs_norm, start=1):
                        sample[_gen_key(args.out_normalized_key, k)] = out_norm if out_norm.strip() else None
                except Exception as e:
                    print(f"❌ Error querying NORMALIZED for sample {idx}: {e}")
                    for k in range(1, args.num_runs + 1):
                        sample[_gen_key(args.out_normalized_key, k)] = None

                if args.sleep > 0:
                    time.sleep(args.sleep)

        processed_count += 1

        if args.save_every > 0 and processed_count % args.save_every == 0:
            try:
                atomic_json_dump(data, args.output_path)
                print(
                    f"💾 Progress saved to {args.output_path} "
                    f"(processed {processed_count} samples in range)"
                )
            except Exception as e:
                print(f"⚠️ Failed to save {args.output_path}: {e}")


    # Final save: write the requested output_path
    atomic_json_dump(data, args.output_path)
    print(f"✨ Saved updated dataset to {args.output_path}")


if __name__ == "__main__":
    main()