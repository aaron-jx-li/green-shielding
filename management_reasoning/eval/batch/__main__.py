#!/usr/bin/env python3
"""CLI for multi-stage diagnosis-eval Vertex Batch (Flash-Lite judges).

Stages:
  extract, unc  — can prepare/submit in parallel (need only target diagnosis text)
  sem, ground   — after extract collect
  aggregate     — after all collects

Examples:
  python -m management_reasoning.eval.batch prepare --stage extract \\
    --suite smoke_n3 --target gemini --arm raw --end_idx 2
  python -m management_reasoning.eval.batch submit --stage extract \\
    --suite smoke_n3 --target gemini --arm raw
  python -m management_reasoning.eval.batch status --stage extract \\
    --suite smoke_n3 --target gemini --arm raw
  python -m management_reasoning.eval.batch collect --stage extract \\
    --suite smoke_n3 --target gemini --arm raw
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from management_reasoning.config import models_config
from management_reasoning.eval.batch.aggregate import aggregate_arm
from management_reasoning.eval.batch.collect import collect_stage
from management_reasoning.eval.batch.format import (
    prepare_extract,
    prepare_ground,
    prepare_sem,
    prepare_unc,
)
from management_reasoning.eval.batch.paths import (
    ARMS,
    DEFAULT_BUCKET,
    DEFAULT_COHORT,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_TRUTH,
    LEGACY_DIAG_COHORT,
    STAGES,
    is_legacy_diag_suite,
    is_legacy_dx_suite,
    local_collected,
    resolve_jobs,
)
from management_reasoning.eval.batch.submit import refresh_stage, submit_stage


def _end(args: argparse.Namespace) -> int:
    if args.end_idx is not None:
        return int(args.end_idx)
    # default: all cohort
    import json

    with open(args.cohort_json, encoding="utf-8") as f:
        return len(json.load(f)) - 1


def cmd_prepare(args: argparse.Namespace) -> None:
    cfg = models_config()
    thinking = args.thinking_level or cfg.get("gemini_judge_thinking_level", "HIGH")
    if (
        (is_legacy_diag_suite(args.suite) or is_legacy_dx_suite(args.suite))
        and args.cohort_json == DEFAULT_COHORT
    ):
        args.cohort_json = LEGACY_DIAG_COHORT
    end = _end(args)
    jobs = resolve_jobs(args.target, args.arm, suite=args.suite)
    for target, arm in jobs:
        if args.stage == "extract":
            path = prepare_extract(
                suite=args.suite,
                target=target,
                arm=arm,
                start_idx=args.start_idx,
                end_idx=end,
                cohort_json=args.cohort_json,
                top_k=args.top_k_dx,
                thinking_level=thinking,
            )
        elif args.stage == "unc":
            path = prepare_unc(
                suite=args.suite,
                target=target,
                arm=arm,
                start_idx=args.start_idx,
                end_idx=end,
                cohort_json=args.cohort_json,
                thinking_level=thinking,
            )
        elif args.stage == "sem":
            extract_path = local_collected(args.suite, target, arm, "extract")
            if not os.path.isfile(extract_path):
                raise SystemExit(f"Need extract collect first: {extract_path}")
            path = prepare_sem(
                suite=args.suite,
                target=target,
                arm=arm,
                extract_collected=extract_path,
                truth_path=args.pxhx_path,
                max_pairs_per_call=args.sem_max_pairs_per_call,
                thinking_level=thinking,
            )
        elif args.stage == "ground":
            extract_path = local_collected(args.suite, target, arm, "extract")
            if not os.path.isfile(extract_path):
                raise SystemExit(f"Need extract collect first: {extract_path}")
            path = prepare_ground(
                suite=args.suite,
                target=target,
                arm=arm,
                extract_collected=extract_path,
                start_idx=args.start_idx,
                end_idx=end,
                cohort_json=args.cohort_json,
                max_grounding_dx=args.max_grounding_dx,
                thinking_level=thinking,
            )
        else:
            raise SystemExit(f"Unknown stage {args.stage}")
        with open(path, encoding="utf-8") as f:
            n = sum(1 for _ in f if _.strip())
        print(f"Prepared {args.suite} {target}/{arm}/{args.stage}: lines={n} -> {path}")


def cmd_submit(args: argparse.Namespace) -> None:
    cfg = models_config()
    model = args.model or cfg.get("gemini_judge", DEFAULT_JUDGE_MODEL)
    jobs = resolve_jobs(args.target, args.arm, suite=args.suite)
    for target, arm in jobs:
        manifest = submit_stage(
            suite=args.suite,
            target=target,
            arm=arm,
            stage=args.stage,
            bucket=args.bucket,
            model=model,
            project=args.project,
            location=args.location,
        )
        print(
            f"Submitted {args.suite} {target}/{arm}/{args.stage}: "
            f"job={manifest.get('job_name')} state={manifest.get('job_state')} "
            f"manifest={manifest.get('manifest_path')}"
        )


def cmd_status(args: argparse.Namespace) -> None:
    jobs = resolve_jobs(args.target, args.arm, suite=args.suite)
    for target, arm in jobs:
        try:
            info = refresh_stage(
                suite=args.suite,
                target=target,
                arm=arm,
                stage=args.stage,
                project=args.project,
                location=args.location,
            )
            print(
                f"{args.suite} {target}/{arm}/{args.stage}: state={info.get('state')} "
                f"job={info.get('job_name')} error={info.get('error')}"
            )
        except FileNotFoundError as e:
            print(f"{args.suite} {target}/{arm}/{args.stage}: {e}")


def cmd_collect(args: argparse.Namespace) -> None:
    jobs = resolve_jobs(args.target, args.arm, suite=args.suite)
    for target, arm in jobs:
        summary = collect_stage(
            suite=args.suite,
            target=target,
            arm=arm,
            stage=args.stage,
            project=args.project,
        )
        n = summary["n"] or 0
        rate = (summary["parse_ok"] / n) if n else 0.0
        print(
            f"Collected {args.suite} {target}/{arm}/{args.stage}: n={n} "
            f"parse_ok={summary['parse_ok']} ({rate:.1%}) errors={summary['errors']} "
            f"out={summary['out_jsonl']}"
        )


def cmd_aggregate(args: argparse.Namespace) -> None:
    if (
        (is_legacy_diag_suite(args.suite) or is_legacy_dx_suite(args.suite))
        and args.cohort_json == DEFAULT_COHORT
    ):
        args.cohort_json = LEGACY_DIAG_COHORT
    jobs = resolve_jobs(args.target, args.arm, suite=args.suite)
    end = _end(args)
    for target, arm in jobs:
        out = aggregate_arm(
            suite=args.suite,
            target=target,
            arm=arm,
            cohort_json=args.cohort_json,
            truth_path=args.pxhx_path,
            top_k_dx=args.top_k_dx,
            start_idx=args.start_idx,
            end_idx=end,
        )
        print(
            f"Aggregated {args.suite} {target}/{arm}: "
            f"n={out['summary'].get('num_with_pxhx')} -> {out['out_path']}"
        )
        print(json_summary(out["summary"]))


def json_summary(summary: dict) -> str:
    import json

    keys = [
        "mean_plausibility",
        "mean_h_coverage",
        "mean_c_coverage",
        "mean_normalized_breadth",
        "mean_support_rate",
        "mean_indirect_inference_rate",
        "uncertainty_rate",
    ]
    slim = {k: summary.get(k) for k in keys}
    return json.dumps(slim, indent=2)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m management_reasoning.eval.batch",
        description="Multi-stage Flash-Lite judge Batch for management_reasoning diagnosis eval",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, need_stage: bool = True) -> None:
        p.add_argument("--suite", default="primary", help="e.g. primary | legacy_diag | smoke_n3")
        p.add_argument("--target", choices=("gemini", "claude"), default=None)
        p.add_argument("--arm", choices=ARMS, default=None)
        if need_stage:
            p.add_argument("--stage", choices=STAGES, required=True)
        p.add_argument("--project", default=None)
        p.add_argument("--location", default="global")
        p.add_argument("--bucket", default=DEFAULT_BUCKET)

    p = sub.add_parser("prepare", help="Build local judge request JSONL")
    common(p)
    p.add_argument("--cohort_json", default=DEFAULT_COHORT)
    p.add_argument("--pxhx_path", default=DEFAULT_TRUTH)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=None)
    p.add_argument("--top_k_dx", type=int, default=8)
    p.add_argument("--max_grounding_dx", type=int, default=8)
    p.add_argument("--sem_max_pairs_per_call", type=int, default=50)
    p.add_argument("--thinking_level", default=None)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("submit", help="Upload + create Batch job")
    common(p)
    p.add_argument("--model", default=None)
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("status", help="Refresh job state")
    common(p)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("collect", help="Download predictions → collected.jsonl")
    common(p)
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("aggregate", help="Combine stages → eval.json metrics")
    common(p, need_stage=False)
    p.add_argument("--cohort_json", default=DEFAULT_COHORT)
    p.add_argument("--pxhx_path", default=DEFAULT_TRUTH)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=None)
    p.add_argument("--top_k_dx", type=int, default=8)
    p.set_defaults(func=cmd_aggregate)

    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
