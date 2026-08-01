#!/usr/bin/env python3
"""CLI: prepare / submit / status / collect for Vertex Batch suites."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from management_reasoning.batch.collect import collect_job
from management_reasoning.batch.format_requests import prepare_local_jsonl
from management_reasoning.batch.paths import (
    DEFAULT_BUCKET,
    DEFAULT_INPUT_PATH,
    LEGACY_DIAG_INPUT_PATH,
    ABLATION_JOBS,
    PRIMARY_JOBS,
    VALID_ARMS,
    input_path_for_arm,
    is_independent_new_neu_suite,
    is_independent_remove_all_suite,
    is_legacy_diag_suite,
    is_legacy_dx_suite,
    model_for_provider,
    suite_jobs,
    suite_tag,
)
from management_reasoning.batch.status import refresh_job
from management_reasoning.batch.submit import submit_job
from management_reasoning.config import models_config
from management_reasoning.prompts import ORDER_VARIANT_IDS


def _load_inputs(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"Expected JSON list in {path}")
    return data


def _resolve_jobs(
    suite: str,
    provider: Optional[str],
    arm: Optional[str],
) -> List[Tuple[str, str]]:
    jobs = suite_jobs(suite)
    # For smoke suites default to full primary grid unless filtered
    if suite.startswith("smoke") and not provider and not arm:
        jobs = list(PRIMARY_JOBS)
    if provider:
        jobs = [(p, a) for p, a in jobs if p == provider]
    if arm:
        jobs = [(p, a) for p, a in jobs if a == arm]
    if not jobs:
        raise SystemExit("No jobs match --provider/--arm filters")
    return jobs


def cmd_prepare(args: argparse.Namespace) -> None:
    default_input = args.input_path
    if (
        (
            is_legacy_diag_suite(args.suite)
            or is_legacy_dx_suite(args.suite)
            or is_independent_remove_all_suite(args.suite)
        )
        and default_input == DEFAULT_INPUT_PATH
    ):
        default_input = LEGACY_DIAG_INPUT_PATH
    jobs = _resolve_jobs(args.suite, args.provider, args.arm)
    # Cache loaded cohorts when multiple jobs share a path.
    loaded: Dict[str, List[Dict[str, Any]]] = {}
    for provider, arm in jobs:
        input_path = input_path_for_arm(args.suite, arm, default_input)
        if input_path not in loaded:
            loaded[input_path] = _load_inputs(input_path)
        samples = loaded[input_path]
        end = args.end_idx if args.end_idx is not None else len(samples) - 1
        path = prepare_local_jsonl(
            samples,
            provider=provider,
            arm=arm,
            suite=args.suite,
            start_idx=args.start_idx,
            end_idx=end,
            include_json_schema=not args.no_json_schema,
        )
        n_samples = end - args.start_idx + 1
        with open(path, "r", encoding="utf-8") as f:
            n_lines = sum(1 for _ in f)
        print(
            f"Prepared {args.suite} {provider}/{arm}: "
            f"samples={n_samples} lines={n_lines} input={input_path} -> {path}"
        )


def cmd_submit(args: argparse.Namespace) -> None:
    models_cfg = models_config()
    jobs = _resolve_jobs(args.suite, args.provider, args.arm)
    for provider, arm in jobs:
        model = args.model or model_for_provider(provider, models_cfg)
        manifest = submit_job(
            provider=provider,
            arm=arm,
            model=model,
            bucket=args.bucket,
            suite=args.suite,
            project=args.project,
            location=args.location,
        )
        print(
            f"Submitted {args.suite} {provider}/{arm}: job={manifest.get('job_name')} "
            f"state={manifest.get('job_state')} manifest={manifest.get('manifest_path')}"
        )


def cmd_status(args: argparse.Namespace) -> None:
    jobs = _resolve_jobs(args.suite, args.provider, args.arm)
    for provider, arm in jobs:
        try:
            info = refresh_job(
                provider=provider,
                arm=arm,
                suite=args.suite,
                project=args.project,
                location=args.location,
            )
            print(
                f"{args.suite} {provider}/{arm}: state={info.get('state')} "
                f"job={info.get('job_name')} error={info.get('error')}"
            )
        except FileNotFoundError as e:
            print(f"{args.suite} {provider}/{arm}: {e}")


def cmd_collect(args: argparse.Namespace) -> None:
    jobs = _resolve_jobs(args.suite, args.provider, args.arm)
    tag = args.tag if args.tag is not None else suite_tag(args.suite)
    for provider, arm in jobs:
        summary = collect_job(
            provider=provider,
            arm=arm,
            suite=args.suite,
            tag=tag,
            project=args.project,
        )
        rate = (summary["parse_ok"] / summary["n"]) if summary["n"] else 0.0
        print(
            f"Collected {args.suite} {provider}/{arm}: n={summary['n']} "
            f"parse_ok={summary['parse_ok']} ({rate:.1%}) "
            f"errors={summary['errors']} out={summary['out_jsonl']}"
        )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m management_reasoning.batch",
        description=(
            "Vertex Batch prepare/submit/status/collect. "
            "Suites: primary | order_ord1|ord2|ord3 | independent | "
            "independent_remove_all | independent_new_neu | legacy_diag | "
            "legacy_dx | smoke_*"
        ),
    )
    sub = ap.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--suite",
            default="primary",
            help=(
                "primary | order_ord1|order_ord2|order_ord3 | independent | "
                "independent_remove_all | independent_new_neu | legacy_diag | "
                "legacy_dx | custom smoke name"
            ),
        )
        p.add_argument("--provider", choices=("gemini", "claude"), default=None)
        p.add_argument("--arm", choices=VALID_ARMS, default=None)
        p.add_argument("--project", default=None)
        p.add_argument("--location", default=None)
        p.add_argument("--bucket", default=DEFAULT_BUCKET)

    p_prep = sub.add_parser("prepare", help="Build local request JSONL files")
    add_common(p_prep)
    p_prep.add_argument("--input_path", default=DEFAULT_INPUT_PATH)
    p_prep.add_argument("--start_idx", type=int, default=0)
    p_prep.add_argument("--end_idx", type=int, default=None, help="Inclusive; default=all")
    p_prep.add_argument(
        "--no_json_schema",
        action="store_true",
        help="Omit responseJsonSchema in Gemini batch requests",
    )
    p_prep.set_defaults(func=cmd_prepare)

    p_sub = sub.add_parser("submit", help="Upload JSONL and create Batch jobs")
    add_common(p_sub)
    p_sub.add_argument("--model", default=None, help="Override model id for filtered provider")
    p_sub.set_defaults(func=cmd_submit)

    p_stat = sub.add_parser("status", help="Refresh and print job states")
    add_common(p_stat)
    p_stat.set_defaults(func=cmd_status)

    p_col = sub.add_parser("collect", help="Download outputs → local responses.jsonl")
    add_common(p_col)
    p_col.add_argument(
        "--tag",
        default=None,
        help="Local out dir suffix (default derived from suite)",
    )
    p_col.set_defaults(func=cmd_collect)

    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)
    # Validate order suites early
    if args.suite.startswith("order_"):
        oid = args.suite[len("order_") :]
        if oid not in ORDER_VARIANT_IDS:
            raise SystemExit(
                f"Unknown suite {args.suite}; expected order_{{|{'|'.join(ORDER_VARIANT_IDS)}}}"
            )
    args.func(args)


if __name__ == "__main__":
    main()
