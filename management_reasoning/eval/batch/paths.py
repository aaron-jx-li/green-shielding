"""Paths for management_reasoning diagnosis-eval Vertex Batch (Flash-Lite judges)."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

DEFAULT_BUCKET = "bin-yu-green-shield-mgmt-reasoning"
DEFAULT_COHORT = "./results/management_reasoning/data/hcm_full_inputs.json"
LEGACY_DIAG_COHORT = "./results/management_reasoning/data/hcm_legacy_diag_inputs.json"
DEFAULT_TRUTH = "./results/HCM-3k/truth/merged_truth_new.json"
DEFAULT_JUDGE_MODEL = "gemini-3.1-flash-lite"

STAGES = ("extract", "unc", "sem", "ground")
TARGETS = ("gemini", "claude")
ARMS = (
    "raw",
    "neutralized",
    "remove_all",
    "ra_new",
    "ct_old",
    "ct_new",
    "format_tone",
    "content_format",
)

PRIMARY_TARGET_JOBS: Tuple[Tuple[str, str], ...] = (
    ("gemini", "raw"),
    ("gemini", "neutralized"),
    ("claude", "raw"),
    ("claude", "neutralized"),
)

LEGACY_DIAG_TARGET_JOBS: Tuple[Tuple[str, str], ...] = (
    ("claude", "raw"),
    ("claude", "remove_all"),
)

# Independent dx smoke: Claude raw + neutralize variants; Gemini raw baseline.
INDEP_DX_SMOKE_JOBS: Tuple[Tuple[str, str], ...] = (
    ("claude", "raw"),
    ("claude", "remove_all"),
    ("claude", "ct_old"),
    ("claude", "ct_new"),
    ("gemini", "raw"),
)

# Legacy free-form dx Flash-Lite: generate arms only (Claude raw/remove_all reuse legacy_diag eval).
LEGACY_DX_EVAL_JOBS: Tuple[Tuple[str, str], ...] = (
    ("claude", "ct_old"),
    ("claude", "ct_new"),
    ("gemini", "raw"),
    ("gemini", "remove_all"),
    ("gemini", "ct_old"),
    ("gemini", "ct_new"),
)

TARGET_RESPONSES: Dict[Tuple[str, str], str] = {
    ("gemini", "raw"): (
        "./results/management_reasoning/responses/vertex/"
        "gemini-3.1-pro-preview/raw_primary_batch/responses.jsonl"
    ),
    ("gemini", "neutralized"): (
        "./results/management_reasoning/responses/vertex/"
        "gemini-3.1-pro-preview/neutralized_primary_batch/responses.jsonl"
    ),
    ("claude", "raw"): (
        "./results/management_reasoning/responses/vertex/"
        "claude-opus-4-5_20251101/raw_primary_batch/responses.jsonl"
    ),
    ("claude", "neutralized"): (
        "./results/management_reasoning/responses/vertex/"
        "claude-opus-4-5_20251101/neutralized_primary_batch/responses.jsonl"
    ),
}

LEGACY_DIAG_TARGET_RESPONSES: Dict[Tuple[str, str], str] = {
    ("claude", "raw"): (
        "./results/HCM-3k/exp_frontier/claude-opus-4-5_20251101/"
        "raw_legacy_batch/responses.jsonl"
    ),
    ("claude", "remove_all"): (
        "./results/HCM-3k/exp_frontier/claude-opus-4-5_20251101/"
        "remove_all_legacy_batch/responses.jsonl"
    ),
}

INDEP_DX_TARGET_RESPONSES: Dict[Tuple[str, str], str] = {
    ("claude", "raw"): (
        "./results/management_reasoning/responses/vertex/"
        "claude-opus-4-5_20251101/raw_independent_batch/responses.jsonl"
    ),
    ("claude", "remove_all"): (
        "./results/management_reasoning/responses/vertex/"
        "claude-opus-4-5_20251101/remove_all_independent_remove_all_batch/"
        "responses.jsonl"
    ),
    ("claude", "ct_old"): (
        "./results/management_reasoning/responses/vertex/"
        "claude-opus-4-5_20251101/ct_old_independent_new_neu_batch/responses.jsonl"
    ),
    ("claude", "ct_new"): (
        "./results/management_reasoning/responses/vertex/"
        "claude-opus-4-5_20251101/ct_new_independent_new_neu_batch/responses.jsonl"
    ),
    ("gemini", "raw"): (
        "./results/management_reasoning/responses/vertex/"
        "gemini-3.1-pro-preview/raw_independent_remove_all_batch/responses.jsonl"
    ),
}

LEGACY_DX_TARGET_RESPONSES: Dict[Tuple[str, str], str] = {
    ("claude", "ct_old"): (
        "./results/HCM-3k/exp_frontier/claude-opus-4-5_20251101/"
        "ct_old_legacy_dx_batch/responses.jsonl"
    ),
    ("claude", "ct_new"): (
        "./results/HCM-3k/exp_frontier/claude-opus-4-5_20251101/"
        "ct_new_legacy_dx_batch/responses.jsonl"
    ),
    ("gemini", "raw"): (
        "./results/HCM-3k/exp_frontier/gemini-3.1-pro-preview/"
        "raw_legacy_dx_batch/responses.jsonl"
    ),
    ("gemini", "remove_all"): (
        "./results/HCM-3k/exp_frontier/gemini-3.1-pro-preview/"
        "remove_all_legacy_dx_batch/responses.jsonl"
    ),
    ("gemini", "ct_old"): (
        "./results/HCM-3k/exp_frontier/gemini-3.1-pro-preview/"
        "ct_old_legacy_dx_batch/responses.jsonl"
    ),
    ("gemini", "ct_new"): (
        "./results/HCM-3k/exp_frontier/gemini-3.1-pro-preview/"
        "ct_new_legacy_dx_batch/responses.jsonl"
    ),
}

# Factor-pair legacy free-form diagnosis (full collect tag).
LEGACY_DX_FACTOR_EVAL_JOBS: Tuple[Tuple[str, str], ...] = (
    ("claude", "format_tone"),
    ("claude", "content_format"),
    ("gemini", "format_tone"),
    ("gemini", "content_format"),
)

LEGACY_DX_FACTOR_TARGET_RESPONSES: Dict[Tuple[str, str], str] = {
    ("claude", "format_tone"): (
        "./results/HCM-3k/exp_frontier/claude-opus-4-5_20251101/"
        "format_tone_legacy_dx_factor_batch/responses.jsonl"
    ),
    ("claude", "content_format"): (
        "./results/HCM-3k/exp_frontier/claude-opus-4-5_20251101/"
        "content_format_legacy_dx_factor_batch/responses.jsonl"
    ),
    ("gemini", "format_tone"): (
        "./results/HCM-3k/exp_frontier/gemini-3.1-pro-preview/"
        "format_tone_legacy_dx_factor_batch/responses.jsonl"
    ),
    ("gemini", "content_format"): (
        "./results/HCM-3k/exp_frontier/gemini-3.1-pro-preview/"
        "content_format_legacy_dx_factor_batch/responses.jsonl"
    ),
}

# gpt-5.2 remove_all (ra_new) — legacy free-form + independent dx Flash-Lite.
RA_NEW_EVAL_JOBS: Tuple[Tuple[str, str], ...] = (
    ("claude", "ra_new"),
    ("gemini", "ra_new"),
)

LEGACY_DX_RA_NEW_TARGET_RESPONSES: Dict[Tuple[str, str], str] = {
    ("claude", "ra_new"): (
        "./results/HCM-3k/exp_frontier/claude-opus-4-5_20251101/"
        "ra_new_legacy_dx_ra_new_batch/responses.jsonl"
    ),
    ("gemini", "ra_new"): (
        "./results/HCM-3k/exp_frontier/gemini-3.1-pro-preview/"
        "ra_new_legacy_dx_ra_new_batch/responses.jsonl"
    ),
}

INDEP_DX_RA_NEW_TARGET_RESPONSES: Dict[Tuple[str, str], str] = {
    ("claude", "ra_new"): (
        "./results/management_reasoning/responses/vertex/"
        "claude-opus-4-5_20251101/ra_new_independent_ra_new_batch/responses.jsonl"
    ),
    ("gemini", "ra_new"): (
        "./results/management_reasoning/responses/vertex/"
        "gemini-3.1-pro-preview/ra_new_independent_ra_new_batch/responses.jsonl"
    ),
}


def is_legacy_diag_suite(suite: str) -> bool:
    return suite == "legacy_diag" or suite.startswith("legacy_diag")


def is_legacy_dx_suite(suite: str) -> bool:
    """Flash-Lite / batch eval suite for legacy free-form dx grid."""
    return suite == "legacy_dx" or suite.startswith("legacy_dx")


def is_legacy_dx_factor_suite(suite: str) -> bool:
    return suite.startswith("legacy_dx_factor")


def is_legacy_dx_ra_new_suite(suite: str) -> bool:
    return suite.startswith("legacy_dx_ra_new")


def is_indep_dx_ra_new_suite(suite: str) -> bool:
    return suite.startswith("indep_dx_ra_new") or suite.startswith(
        "independent_dx_ra_new"
    )


def is_independent_dx_suite(suite: str) -> bool:
    return (
        suite.startswith("indep_dx")
        or suite.startswith("independent_dx")
        or suite == "independent_dx"
    )


def question_filter_for(suite: str) -> Optional[str]:
    """Independent JSONL has 5 rows/sample; Flash-Lite scores dx only."""
    if is_independent_dx_suite(suite):
        return "dx"
    return None


def _frontier_model_dir(target: str) -> str:
    if target == "claude":
        return "claude-opus-4-5_20251101"
    return "gemini-3.1-pro-preview"


def response_path_for(suite: str, target: str, arm: str) -> str:
    if is_legacy_diag_suite(suite):
        key = (target, arm)
        if key not in LEGACY_DIAG_TARGET_RESPONSES:
            raise KeyError(f"No legacy_diag responses for {target}/{arm}")
        return LEGACY_DIAG_TARGET_RESPONSES[key]
    if is_legacy_dx_ra_new_suite(suite):
        if suite in ("legacy_dx_ra_new", "legacy_dx_ra_new_batch"):
            key = (target, arm)
            if key not in LEGACY_DX_RA_NEW_TARGET_RESPONSES:
                raise KeyError(f"No legacy_dx_ra_new responses for {target}/{arm}")
            return LEGACY_DX_RA_NEW_TARGET_RESPONSES[key]
        tag = suite if suite.endswith("_batch") else f"{suite}_batch"
        return (
            f"./results/HCM-3k/exp_frontier/{_frontier_model_dir(target)}/"
            f"{arm}_{tag}/responses.jsonl"
        )
    if is_legacy_dx_factor_suite(suite):
        if suite == "legacy_dx_factor" or suite == "legacy_dx_factor_batch":
            key = (target, arm)
            if key not in LEGACY_DX_FACTOR_TARGET_RESPONSES:
                raise KeyError(f"No legacy_dx_factor responses for {target}/{arm}")
            return LEGACY_DX_FACTOR_TARGET_RESPONSES[key]
        # Smoke / other variants: {arm}_{suite}_batch
        tag = suite if suite.endswith("_batch") else f"{suite}_batch"
        return (
            f"./results/HCM-3k/exp_frontier/{_frontier_model_dir(target)}/"
            f"{arm}_{tag}/responses.jsonl"
        )
    if is_legacy_dx_suite(suite):
        key = (target, arm)
        if key not in LEGACY_DX_TARGET_RESPONSES:
            raise KeyError(f"No legacy_dx responses for {target}/{arm}")
        return LEGACY_DX_TARGET_RESPONSES[key]
    if is_indep_dx_ra_new_suite(suite):
        if suite in ("indep_dx_ra_new", "independent_dx_ra_new"):
            key = (target, arm)
            if key not in INDEP_DX_RA_NEW_TARGET_RESPONSES:
                raise KeyError(f"No indep_dx_ra_new responses for {target}/{arm}")
            return INDEP_DX_RA_NEW_TARGET_RESPONSES[key]
        # Smoke: responses still under independent_ra_new*_batch collect tag
        tag = "independent_ra_new_batch"
        if "smoke" in suite:
            # Map indep_dx_ra_new_smoke_n10 → independent_ra_new_smoke_n10_batch
            smoke = suite.replace("indep_dx_ra_new", "independent_ra_new").replace(
                "independent_dx_ra_new", "independent_ra_new"
            )
            tag = smoke if smoke.endswith("_batch") else f"{smoke}_batch"
        return (
            f"./results/management_reasoning/responses/vertex/"
            f"{_frontier_model_dir(target)}/{arm}_{tag}/responses.jsonl"
        )
    if is_independent_dx_suite(suite):
        key = (target, arm)
        if key not in INDEP_DX_TARGET_RESPONSES:
            raise KeyError(f"No independent dx responses for {target}/{arm}")
        return INDEP_DX_TARGET_RESPONSES[key]
    return TARGET_RESPONSES[(target, arm)]


def answer_mode_for(suite: str) -> str:
    """``diagnosis`` (MR JSON field) or ``full_response`` (paper free-form)."""
    if is_legacy_diag_suite(suite) or is_legacy_dx_suite(suite):
        return "full_response"
    return "diagnosis"


def local_root(suite: str, target: str, arm: str, stage: str) -> str:
    return os.path.join(
        "./results/management_reasoning/eval/batch",
        suite,
        target,
        arm,
        stage,
    )


def local_requests(suite: str, target: str, arm: str, stage: str) -> str:
    return os.path.join(local_root(suite, target, arm, stage), "requests.jsonl")


def local_manifest(suite: str, target: str, arm: str, stage: str) -> str:
    return os.path.join(local_root(suite, target, arm, stage), "job_manifest.json")


def local_pair_index(suite: str, target: str, arm: str) -> str:
    return os.path.join(local_root(suite, target, arm, "sem"), "pair_index.jsonl")


def local_collected(suite: str, target: str, arm: str, stage: str) -> str:
    return os.path.join(local_root(suite, target, arm, stage), "collected.jsonl")


def local_eval_out(suite: str, target: str, arm: str) -> str:
    return os.path.join(
        "./results/management_reasoning/eval",
        "gemini-3.1-flash-lite",
        f"{target}_{arm}_{suite}",
        "eval.json",
    )


def gcs_input_uri(bucket: str, suite: str, target: str, arm: str, stage: str) -> str:
    return f"gs://{bucket}/eval_batch/{suite}/{target}/{arm}/{stage}/input/requests.jsonl"


def gcs_output_prefix(bucket: str, suite: str, target: str, arm: str, stage: str) -> str:
    return f"gs://{bucket}/eval_batch/{suite}/{target}/{arm}/{stage}/output/"


def make_custom_id(
    stage: str,
    target: str,
    arm: str,
    sample_id: int,
    *,
    chunk: Optional[int] = None,
) -> str:
    base = f"j_{stage}_{target}_{arm}_{sample_id}"
    if chunk is not None:
        return f"{base}_c{chunk}"
    return base


def parse_custom_id(cid: str) -> dict:
    if not cid.startswith("j_"):
        raise ValueError(f"Bad judge custom_id: {cid}")
    rest = cid[2:]
    stage = None
    for s in STAGES:
        prefix = s + "_"
        if rest.startswith(prefix):
            stage = s
            rest = rest[len(prefix) :]
            break
    if stage is None:
        raise ValueError(f"Bad judge custom_id stage: {cid}")

    target = None
    for t in TARGETS:
        prefix = t + "_"
        if rest.startswith(prefix):
            target = t
            rest = rest[len(prefix) :]
            break
    if target is None:
        raise ValueError(f"Bad judge custom_id target: {cid}")

    arm = None
    # Longest tokens first (content_format/format_tone before content-like; ct_* before raw).
    for a in (
        "content_format",
        "format_tone",
        "remove_all",
        "neutralized",
        "ra_new",
        "ct_old",
        "ct_new",
        "raw",
    ):
        prefix = a + "_"
        if rest.startswith(prefix):
            arm = a
            rest = rest[len(prefix) :]
            break
    if arm is None:
        raise ValueError(f"Bad judge custom_id arm: {cid}")

    chunk = None
    if "_c" in rest:
        sid_s, _, chunk_s = rest.rpartition("_c")
        return {
            "stage": stage,
            "target": target,
            "arm": arm,
            "sample_id": int(sid_s),
            "chunk": int(chunk_s),
        }
    return {
        "stage": stage,
        "target": target,
        "arm": arm,
        "sample_id": int(rest),
        "chunk": None,
    }


def resolve_jobs(
    target: Optional[str],
    arm: Optional[str],
    *,
    suite: str = "primary",
) -> List[Tuple[str, str]]:
    if is_legacy_diag_suite(suite):
        jobs = list(LEGACY_DIAG_TARGET_JOBS)
    elif is_legacy_dx_ra_new_suite(suite):
        jobs = list(RA_NEW_EVAL_JOBS)
    elif is_indep_dx_ra_new_suite(suite):
        jobs = list(RA_NEW_EVAL_JOBS)
    elif is_legacy_dx_factor_suite(suite):
        jobs = list(LEGACY_DX_FACTOR_EVAL_JOBS)
    elif is_legacy_dx_suite(suite):
        jobs = list(LEGACY_DX_EVAL_JOBS)
    elif is_independent_dx_suite(suite):
        jobs = list(INDEP_DX_SMOKE_JOBS)
    else:
        jobs = list(PRIMARY_TARGET_JOBS)
    if target:
        jobs = [(t, a) for t, a in jobs if t == target]
    if arm:
        jobs = [(t, a) for t, a in jobs if a == arm]
    if not jobs:
        raise ValueError("No jobs match --target/--arm")
    return jobs
