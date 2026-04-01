from __future__ import annotations

import json
import time
from typing import Any, Callable, List, Optional, Protocol

from tqdm import tqdm

from normalization.io import atomic_json_dump


class _InferenceArgs(Protocol):
    input_path: str
    output_path: str
    raw_key: str
    normalized_key: str
    out_raw_key: str
    out_normalized_key: str
    num_runs: int
    instruction: str
    model: str
    temperature: float
    max_tokens: Optional[int]
    seed: Optional[int]
    start_idx: Optional[int]
    end_idx: Optional[int]
    sleep: float
    skip_existing: bool
    save_every: int


def _nonempty(x: Any) -> bool:
    return isinstance(x, str) and x.strip() != ""


def _gen_key(base: str, k: int) -> str:
    return f"{base}_{k}"


def _all_runs_present(sample: dict, base_key: str, n: int) -> bool:
    return all(_nonempty(sample.get(_gen_key(base_key, k), "")) for k in range(1, n + 1))


def _pop_run_keys(sample: dict, base_key: str, n: int) -> None:
    for k in range(1, n + 1):
        sample.pop(_gen_key(base_key, k), None)


def run_inference_main(
    args: _InferenceArgs,
    *,
    client: Any,
    query_fn: Callable[..., List[str]],
    respect_mode: bool,
) -> None:
    """
    Shared loop: load JSON list, optionally clear run keys outside range, query model per sample.

    ``query_fn`` is called as ``query_fn(client, input_text, **query_kw)`` and must return ``num_runs`` strings.

    When ``respect_mode`` is True, only query raw/normalized branches allowed by ``args.mode``.
    When False, query any branch with non-empty text (multi-provider script behavior).
    """
    if args.num_runs < 1:
        raise ValueError("❌ --num_runs must be >= 1")

    with open(args.input_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("❌ Input JSON must be a list of samples (each a dict).")

    total = len(data)
    print(f"Loaded {total} samples from {args.input_path}.")

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

    for i, sample in enumerate(data):
        if not isinstance(sample, dict):
            continue
        if i not in processed_set:
            _pop_run_keys(sample, args.out_raw_key, args.num_runs)
            _pop_run_keys(sample, args.out_normalized_key, args.num_runs)

    mode = getattr(args, "mode", "both")
    processed_count = 0

    query_kw = {
        "model": args.model,
        "instruction": args.instruction,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "num_runs": args.num_runs,
        "seed": args.seed,
    }

    for idx in tqdm(sorted(processed_set)):
        if idx < 0 or idx >= total:
            print(f"❌ Index {idx} is out of bounds. Skipping.")
            continue

        sample = data[idx]
        if not isinstance(sample, dict):
            print(f"❌ Sample {idx} is not a JSON object/dict. Skipping.")
            continue

        if respect_mode:
            raw_text = sample.get(args.raw_key, "") if mode in ("both", "raw") else ""
            norm_text = sample.get(args.normalized_key, "") if mode in ("both", "normalized") else ""
        else:
            raw_text = sample.get(args.raw_key, "")
            norm_text = sample.get(args.normalized_key, "")

        if raw_text:
            if not (args.skip_existing and _all_runs_present(sample, args.out_raw_key, args.num_runs)):
                try:
                    outs_raw = query_fn(client, raw_text, **query_kw)
                    for k, out_raw in enumerate(outs_raw, start=1):
                        sample[_gen_key(args.out_raw_key, k)] = out_raw if out_raw.strip() else None
                except Exception as e:
                    print(f"❌ Error querying RAW for sample {idx}: {e}")
                    for k in range(1, args.num_runs + 1):
                        sample[_gen_key(args.out_raw_key, k)] = None

                if args.sleep > 0:
                    time.sleep(args.sleep)

        if norm_text:
            if not (args.skip_existing and _all_runs_present(sample, args.out_normalized_key, args.num_runs)):
                try:
                    outs_norm = query_fn(client, norm_text, **query_kw)
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
                print(f"💾 Progress saved to {args.output_path} (processed {processed_count} samples in range)")
            except Exception as e:
                print(f"⚠️ Failed to save {args.output_path}: {e}")

    atomic_json_dump(data, args.output_path)
    print(f"✨ Saved updated dataset to {args.output_path}")
