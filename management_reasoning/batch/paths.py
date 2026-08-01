"""Shared path helpers for Vertex Batch suites (primary, order, independent, legacy_diag)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from management_reasoning.prompts import ORDER_VARIANT_IDS, QUESTION_IDS

DEFAULT_BUCKET = "bin-yu-green-shield-mgmt-reasoning"
DEFAULT_INPUT_PATH = "./results/management_reasoning/data/hcm_full_inputs.json"
LEGACY_DIAG_INPUT_PATH = "./results/management_reasoning/data/hcm_legacy_diag_inputs.json"
LOCAL_MANIFEST_ROOT = "./results/management_reasoning/batch/primary"

# Arms used across suites. ``remove_all`` = paper format collapse;
# ``ct_old`` / ``ct_new`` = gpt-5.2 content+tone variants from results/new_neu/;
# ``ra_new`` = gpt-5.2 content+format+tone (new-model remove_all);
# ``format_tone`` / ``content_format`` = paper 2-factor pairs.
VALID_ARMS = (
    "raw",
    "neutralized",
    "remove_all",
    "ra_new",
    "ct_old",
    "ct_new",
    "format_tone",
    "content_format",
)

PRIMARY_JOBS: Tuple[Tuple[str, str], ...] = (
    ("gemini", "raw"),
    ("gemini", "neutralized"),
    ("claude", "raw"),
    ("claude", "neutralized"),
)

# Default ablation grid: Claude both arms (cost-efficient); override via CLI.
ABLATION_JOBS: Tuple[Tuple[str, str], ...] = (
    ("claude", "raw"),
    ("claude", "neutralized"),
)

# Paper diagnosis-only protocol on frontier Claude (model isolation).
LEGACY_DIAG_JOBS: Tuple[Tuple[str, str], ...] = (
    ("claude", "raw"),
    ("claude", "remove_all"),
)

# Legacy free-form dx grid (Claude raw/remove_all reused from legacy_diag).
LEGACY_DX_JOBS: Tuple[Tuple[str, str], ...] = (
    ("claude", "raw"),
    ("claude", "remove_all"),
    ("claude", "ct_old"),
    ("claude", "ct_new"),
    ("gemini", "raw"),
    ("gemini", "remove_all"),
    ("gemini", "ct_old"),
    ("gemini", "ct_new"),
)

# Arms that still need generation (Claude raw/remove_all already collected).
LEGACY_DX_GENERATE_JOBS: Tuple[Tuple[str, str], ...] = (
    ("claude", "ct_old"),
    ("claude", "ct_new"),
    ("gemini", "raw"),
    ("gemini", "remove_all"),
    ("gemini", "ct_old"),
    ("gemini", "ct_new"),
)

# Independent MR card × remove_all user text (+ Gemini raw; Claude raw reused).
INDEPENDENT_REMOVE_ALL_JOBS: Tuple[Tuple[str, str], ...] = (
    ("claude", "remove_all"),
    ("gemini", "raw"),
    ("gemini", "remove_all"),
)

# Independent MR card × new_neu content+tone (raw reused for both models).
INDEPENDENT_NEW_NEU_JOBS: Tuple[Tuple[str, str], ...] = (
    ("claude", "ct_old"),
    ("claude", "ct_new"),
    ("gemini", "ct_old"),
    ("gemini", "ct_new"),
)

NEW_NEU_INPUT_BY_ARM: Dict[str, str] = {
    "ct_old": "./results/management_reasoning/data/hcm_new_neu_ct_old_inputs.json",
    "ct_new": "./results/management_reasoning/data/hcm_new_neu_ct_new_inputs.json",
    "ra_new": "./results/management_reasoning/data/hcm_new_neu_ra_new_inputs.json",
}

NEW_NEU_SOURCE_BY_ARM: Dict[str, str] = {
    "ct_old": "./results/new_neu/gpt-5.2_old_remove_content_tone.json",
    "ct_new": "./results/new_neu/gpt-5.2_new_remove_content_tone.json",
    "ra_new": "./results/new_neu/gpt-5.2_old_remove_content_format_tone.json",
}

NEW_NEU_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "ct_old": ("content", "tone"),
    "ct_new": ("content", "tone"),
    "ra_new": ("content", "format", "tone"),
}

# gpt-5.2 remove_all (content+format+tone) × Claude/Gemini.
RA_NEW_JOBS: Tuple[Tuple[str, str], ...] = (
    ("claude", "ra_new"),
    ("gemini", "ra_new"),
)

# Paper factor-pair ablations (old neutralize model/phrasing).
FACTOR_PAIR_JOBS: Tuple[Tuple[str, str], ...] = (
    ("claude", "format_tone"),
    ("claude", "content_format"),
    ("gemini", "format_tone"),
    ("gemini", "content_format"),
)

FACTOR_PAIR_INPUT_BY_ARM: Dict[str, str] = {
    "format_tone": "./results/management_reasoning/data/hcm_format_tone_inputs.json",
    "content_format": "./results/management_reasoning/data/hcm_content_format_inputs.json",
}

FACTOR_PAIR_SOURCE_BY_ARM: Dict[str, str] = {
    "format_tone": "./results/HCM-3k/neutralized_prompts/remove_format_tone.json",
    "content_format": "./results/HCM-3k/neutralized_prompts/remove_content_format.json",
}

FACTOR_PAIR_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "format_tone": ("format", "tone"),
    "content_format": ("content", "format"),
}

LEGACY_DIAG_INSTRUCTION = (
    "You are a helpful medical assistant. "
    "Please answer the medical question given by the user. "
    "Be concise, accurate, and clinically realistic."
)

LEGACY_DIAG_TEMPERATURE = 0.7


@dataclass(frozen=True)
class CustomIdParts:
    provider: str
    arm: str
    sample_id: int
    order_id: Optional[str] = None
    question_id: Optional[str] = None


def model_for_provider(provider: str, models_cfg: dict) -> str:
    if provider == "claude":
        return models_cfg.get("claude_target", "claude-opus-4-5@20251101")
    return models_cfg.get("gemini_target", "gemini-3.1-pro-preview")


def model_dir(model: str) -> str:
    return model.replace("@", "_")


def custom_id(
    provider: str,
    arm: str,
    sample_id: int,
    *,
    order_id: Optional[str] = None,
    question_id: Optional[str] = None,
) -> str:
    """
    Encode join key.

    - primary: ``{provider}-{arm}-{sample_id}``
    - order: ``{provider}-{arm}-{order_id}-{sample_id}``
    - independent: ``{provider}-{arm}-{question_id}-{sample_id}``
    """
    if order_id and question_id:
        raise ValueError("custom_id cannot set both order_id and question_id")
    if order_id:
        return f"{provider}-{arm}-{order_id}-{sample_id}"
    if question_id:
        return f"{provider}-{arm}-{question_id}-{sample_id}"
    return f"{provider}-{arm}-{sample_id}"


def parse_custom_id(cid: str) -> CustomIdParts:
    parts = cid.split("-")
    if len(parts) < 3:
        raise ValueError(f"Bad custom_id: {cid}")
    provider = parts[0]
    sample_id = int(parts[-1])
    mid = parts[1:-1]
    if not mid:
        raise ValueError(f"Bad custom_id: {cid}")
    # Longest / multi-token arm names first.
    if mid[0] == "remove_all":
        arm = "remove_all"
        rest = mid[1:]
    elif mid[0] == "neutralized":
        arm = "neutralized"
        rest = mid[1:]
    elif mid[0] == "content_format":
        arm = "content_format"
        rest = mid[1:]
    elif mid[0] == "format_tone":
        arm = "format_tone"
        rest = mid[1:]
    elif mid[0] == "ra_new":
        arm = "ra_new"
        rest = mid[1:]
    elif mid[0] in ("ct_old", "ct_new"):
        arm = mid[0]
        rest = mid[1:]
    else:
        arm = mid[0]
        rest = mid[1:]
    if provider not in ("gemini", "claude") or arm not in VALID_ARMS:
        raise ValueError(f"Bad custom_id: {cid}")
    order_id = None
    question_id = None
    if rest:
        token = rest[0]
        if token in ORDER_VARIANT_IDS:
            order_id = token
        elif token in QUESTION_IDS:
            question_id = token
        else:
            raise ValueError(f"Bad custom_id: {cid}")
        if len(rest) > 1:
            raise ValueError(f"Bad custom_id: {cid}")
    return CustomIdParts(
        provider=provider,
        arm=arm,
        sample_id=sample_id,
        order_id=order_id,
        question_id=question_id,
    )


def suite_jobs(suite: str) -> List[Tuple[str, str]]:
    """Provider/arm pairs for a suite name."""
    if suite == "legacy_diag" or suite.startswith("legacy_diag"):
        return list(LEGACY_DIAG_JOBS)
    if suite.startswith("legacy_dx_ra_new") or suite.startswith("independent_ra_new"):
        return list(RA_NEW_JOBS)
    if suite.startswith("legacy_dx_factor") or suite.startswith("independent_factor"):
        return list(FACTOR_PAIR_JOBS)
    # Smoke only generates missing arms (Claude raw/remove_all reused).
    if suite.startswith("legacy_dx_smoke"):
        return list(LEGACY_DX_GENERATE_JOBS)
    if suite == "legacy_dx" or suite.startswith("legacy_dx"):
        return list(LEGACY_DX_JOBS)
    if suite == "independent_new_neu" or suite.startswith("independent_new_neu"):
        return list(INDEPENDENT_NEW_NEU_JOBS)
    if suite == "independent_remove_all" or suite.startswith("independent_remove_all"):
        return list(INDEPENDENT_REMOVE_ALL_JOBS)
    if suite == "primary" or suite.startswith("smoke"):
        # smoke_* uses same provider/arm filters as primary grid by default
        return list(PRIMARY_JOBS)
    if suite.startswith("order_") or suite == "independent":
        return list(ABLATION_JOBS)
    # Unknown suites: allow explicit --provider/--arm only
    return list(PRIMARY_JOBS)


def suite_tag(suite: str) -> str:
    """Local responses directory suffix."""
    if suite == "primary":
        return "primary_batch"
    if suite == "legacy_diag" or suite.startswith("legacy_diag"):
        return "legacy_batch"
    if suite == "legacy_dx" or suite.startswith("legacy_dx"):
        # Keep smoke suite names distinct (legacy_dx_smoke_n10_batch).
        if suite == "legacy_dx":
            return "legacy_dx_batch"
        return f"{suite}_batch"
    return f"{suite}_batch"


def is_legacy_diag_suite(suite: str) -> bool:
    return suite == "legacy_diag" or suite.startswith("legacy_diag")


def is_legacy_dx_suite(suite: str) -> bool:
    return suite == "legacy_dx" or suite.startswith("legacy_dx")


def is_legacy_freeform_suite(suite: str) -> bool:
    """legacy_diag or legacy_dx (paper system + free-form)."""
    return is_legacy_diag_suite(suite) or is_legacy_dx_suite(suite)


def is_independent_remove_all_suite(suite: str) -> bool:
    return suite == "independent_remove_all" or suite.startswith(
        "independent_remove_all"
    )


def is_independent_new_neu_suite(suite: str) -> bool:
    return suite == "independent_new_neu" or suite.startswith("independent_new_neu")


def is_independent_factor_suite(suite: str) -> bool:
    return suite == "independent_factor" or suite.startswith("independent_factor")


def is_legacy_dx_factor_suite(suite: str) -> bool:
    return suite.startswith("legacy_dx_factor")


def is_legacy_dx_ra_new_suite(suite: str) -> bool:
    return suite.startswith("legacy_dx_ra_new")


def is_independent_ra_new_suite(suite: str) -> bool:
    return suite == "independent_ra_new" or suite.startswith("independent_ra_new")


def input_path_for_arm(suite: str, arm: str, default: str) -> str:
    """Resolve cohort JSON for a prepare job (new_neu / legacy_dx / factor arms)."""
    if arm in FACTOR_PAIR_INPUT_BY_ARM and (
        is_legacy_dx_factor_suite(suite) or is_independent_factor_suite(suite)
    ):
        return FACTOR_PAIR_INPUT_BY_ARM[arm]
    if arm == "ra_new" and (
        is_legacy_dx_ra_new_suite(suite) or is_independent_ra_new_suite(suite)
    ):
        return NEW_NEU_INPUT_BY_ARM["ra_new"]
    if arm in NEW_NEU_INPUT_BY_ARM and (
        is_independent_new_neu_suite(suite) or is_legacy_dx_suite(suite)
    ):
        return NEW_NEU_INPUT_BY_ARM[arm]
    if is_legacy_dx_suite(suite) and arm in ("raw", "remove_all"):
        return LEGACY_DIAG_INPUT_PATH
    return default


def gcs_input_uri(bucket: str, provider: str, arm: str, *, suite: str = "primary") -> str:
    return f"gs://{bucket}/batch/{suite}/{provider}/{arm}/input/requests.jsonl"


def gcs_output_prefix(bucket: str, provider: str, arm: str, *, suite: str = "primary") -> str:
    return f"gs://{bucket}/batch/{suite}/{provider}/{arm}/output/"


def local_manifest_dir(provider: str, arm: str, *, suite: str = "primary") -> str:
    root = (
        LOCAL_MANIFEST_ROOT
        if suite == "primary"
        else f"./results/management_reasoning/batch/{suite}"
    )
    return os.path.join(root, provider, arm)


def local_request_jsonl(provider: str, arm: str, *, suite: str = "primary") -> str:
    return os.path.join(local_manifest_dir(provider, arm, suite=suite), "requests.jsonl")


def local_manifest_path(provider: str, arm: str, *, suite: str = "primary") -> str:
    return os.path.join(local_manifest_dir(provider, arm, suite=suite), "job_manifest.json")


def local_responses_dir(
    model: str,
    arm: str,
    *,
    tag: str = "primary_batch",
    suite: Optional[str] = None,
) -> str:
    """Default MR Vertex responses path; legacy free-form under HCM-3k/exp_frontier."""
    if suite is not None and is_legacy_freeform_suite(suite):
        return (
            f"./results/HCM-3k/exp_frontier/{model_dir(model)}/"
            f"{arm}_{tag}"
        )
    return f"./results/management_reasoning/responses/vertex/{model_dir(model)}/{arm}_{tag}"


def batch_model_resource(provider: str, model: str) -> str:
    """Model string for google.genai batches.create."""
    if provider == "claude":
        if model.startswith("publishers/"):
            return model
        return f"publishers/anthropic/models/{model}"
    if model.startswith("publishers/"):
        return model
    return model


def resolve_order_for_suite(suite: str) -> Optional[Tuple[str, ...]]:
    """Return question order tuple for order_* suites; None for default/primary."""
    from management_reasoning.prompts import ORDER_VARIANTS

    if suite.startswith("order_"):
        oid = suite[len("order_") :]
        if oid not in ORDER_VARIANTS:
            raise ValueError(f"Unknown order suite {suite}; expected order_ord1|ord2|ord3")
        return ORDER_VARIANTS[oid]
    return None


def suite_mode(suite: str) -> str:
    """``multiask`` | ``independent`` | ``legacy_freeform``."""
    if suite == "independent" or suite.startswith("independent"):
        return "independent"
    if is_legacy_freeform_suite(suite):
        return "legacy_freeform"
    return "multiask"


def suite_order_id(suite: str) -> Optional[str]:
    if suite.startswith("order_"):
        return suite[len("order_") :]
    return None
