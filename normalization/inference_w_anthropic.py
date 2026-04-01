"""Query OpenAI or Anthropic for JSON samples using raw + normalized inputs."""

import argparse
import os
import sys

from openai import OpenAI

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from normalization.inference.providers import detect_provider, make_anthropic_client, query_multi_provider
from normalization.inference.runner import run_inference_main

DEFAULT_INSTRUCTION = (
    "You are a helpful medical assistant. "
    "Please answer the medical question given by the user. "
    "Be concise, accurate, and clinically realistic."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query an OpenAI or Anthropic model for JSON samples using raw + normalized inputs."
    )

    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSON file (list of samples).")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save the output JSON (new file) with model responses added.")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="Model name, e.g., gpt-4o, claude-3-5-sonnet, etc.")
    parser.add_argument("--provider", type=str, default=None, choices=["openai", "anthropic"], help="Explicit API provider. If not set, auto-detects from model name.")
    parser.add_argument("--anthropic_api_key", type=str, default=None, help="Anthropic API key (else ANTHROPIC_API_KEY).")
    parser.add_argument("--raw_key", type=str, default="raw_input", help='JSON key for the raw patient text (default: "raw_input").')
    parser.add_argument("--normalized_key", type=str, default="normalized_prompt", help='JSON key for the normalized prompt (default: "normalized_prompt").')
    parser.add_argument("--out_raw_key", type=str, default="model_response_raw", help='Base JSON key for raw input runs (default: "model_response_raw").')
    parser.add_argument("--out_normalized_key", type=str, default="model_response_converted", help='Base JSON key for normalized runs (default: "model_response_converted").')
    parser.add_argument("--num_runs", type=int, default=5, help="Number of independent runs per input. Keys will be suffixed _1.._n.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducibility (if supported).")
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION, help="Developer instruction to send to the model.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (ignored for some models).")
    parser.add_argument("--max_tokens", type=int, default=None, help="Max output tokens.")
    parser.add_argument("--start_idx", type=int, default=None, help="Start index (inclusive).")
    parser.add_argument("--end_idx", type=int, default=None, help="End index (inclusive).")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between requests (default: 0.0).")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip querying if all output keys for all runs already exist and are non-empty.",
    )
    parser.add_argument("--save_every", type=int, default=50, help="Checkpoint every N samples (<=0 disables).")

    return parser.parse_args()


def _get_provider(args: argparse.Namespace) -> str:
    if args.provider is not None:
        return args.provider
    return detect_provider(args.model)


def main() -> None:
    args = parse_args()
    provider = _get_provider(args)
    if provider == "anthropic":
        client = make_anthropic_client(args.anthropic_api_key)
    else:
        client = OpenAI()

    def _query(cli: object, input_text: str, **kw: object) -> list[str]:
        return query_multi_provider(
            cli,  # type: ignore[arg-type]
            provider,
            input_text,
            model=str(kw["model"]),
            instruction=str(kw["instruction"]),
            temperature=float(kw["temperature"]),
            max_tokens=kw["max_tokens"],  # type: ignore[arg-type]
            num_runs=int(kw["num_runs"]),
            seed=kw["seed"],  # type: ignore[arg-type]
        )

    run_inference_main(args, client=client, query_fn=_query, respect_mode=False)


if __name__ == "__main__":
    main()
