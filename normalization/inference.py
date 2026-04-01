"""Query an OpenAI model for JSON samples using raw + normalized inputs (OpenAI-only, legacy kwargs)."""

import argparse
import os
import sys

from openai import OpenAI

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from normalization.inference.providers import query_openai_responses_system
from normalization.inference.runner import run_inference_main

DEFAULT_INSTRUCTION = (
    "You are a helpful medical assistant. "
    "Please answer the medical question given by the user. "
    "Be concise, accurate, and clinically realistic."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query an OpenAI model for JSON samples using raw + normalized inputs."
    )

    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSON file (list of samples).")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save the output JSON (new file) with model responses added.")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="Model name, e.g., gpt-4o, gpt-4o-mini, gpt-5.1, etc.")
    parser.add_argument("--raw_key", type=str, default="raw_input", help='JSON key for the raw patient text (default: "raw_input").')
    parser.add_argument("--normalized_key", type=str, default="neutralized_prompt", help='JSON key for the normalized prompt (default: "normalized_prompt").')
    parser.add_argument("--out_raw_key", type=str, default="model_response_raw", help='Base JSON key for raw input runs (default: "model_response_raw").')
    parser.add_argument("--out_normalized_key", type=str, default="model_response_neutralized", help='Base JSON key for normalized runs (default: "model_response_neutralized").')
    parser.add_argument("--num_runs", type=int, default=5, help="Number of independent runs per input (default: 1). Keys will be suffixed _1.._n.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducibility (if supported by the model).")
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION, help="Developer instruction to send to the model.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (ignored for some models).")
    parser.add_argument("--max_tokens", type=int, default=None, help="Max output tokens.")
    parser.add_argument("--start_idx", type=int, default=None, help="Start index (inclusive).")
    parser.add_argument("--end_idx", type=int, default=None, help="End index (inclusive).")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between requests (default: 0.0).")
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["both", "raw", "normalized"],
        help='Which inputs to query: "both" (default), "raw" only, or "normalized" only.',
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip querying if all output keys for all runs already exist and are non-empty.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=50,
        help="Save a checkpoint every N processed samples (default: 50). Set <=0 to disable.",
    )

    return parser.parse_args()


def _query(client: OpenAI, input_text: str, **kw: object) -> list[str]:
    return query_openai_responses_system(
        client,
        input_text,
        model=str(kw["model"]),
        instruction=str(kw["instruction"]),
        temperature=float(kw["temperature"]),
        max_tokens=kw["max_tokens"],  # type: ignore[arg-type]
        num_runs=int(kw["num_runs"]),
        seed=kw["seed"],  # type: ignore[arg-type]
    )


def main() -> None:
    args = parse_args()
    client = OpenAI()
    run_inference_main(args, client=client, query_fn=_query, respect_mode=True)


if __name__ == "__main__":
    main()
