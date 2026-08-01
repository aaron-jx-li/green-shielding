"""Submit / status for diagnosis-eval judge Batch jobs (Gemini Flash-Lite)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from management_reasoning.batch import gcs_io
from management_reasoning.clients.vertex import get_location, get_project, make_vertex_client
from management_reasoning.eval.batch.paths import (
    DEFAULT_JUDGE_MODEL,
    gcs_input_uri,
    gcs_output_prefix,
    local_manifest,
    local_requests,
)


def _write_manifest(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def submit_stage(
    *,
    suite: str,
    target: str,
    arm: str,
    stage: str,
    bucket: str,
    model: str = DEFAULT_JUDGE_MODEL,
    project: Optional[str] = None,
    location: Optional[str] = None,
) -> Dict[str, Any]:
    proj = get_project(project)
    loc = get_location(location or "global")
    local_jsonl = local_requests(suite, target, arm, stage)
    if not os.path.isfile(local_jsonl):
        raise FileNotFoundError(f"Missing {local_jsonl}. Run prepare for this stage first.")
    # Empty sem jobs (all fuzzy-matched) — skip submit
    with open(local_jsonl, "r", encoding="utf-8") as f:
        n_lines = sum(1 for line in f if line.strip())
    if n_lines == 0:
        manifest = {
            "suite": suite,
            "target": target,
            "arm": arm,
            "stage": stage,
            "model": model,
            "project": proj,
            "location": loc,
            "skipped_empty": True,
            "job_name": None,
            "job_state": "SKIPPED_EMPTY",
        }
        path = local_manifest(suite, target, arm, stage)
        _write_manifest(path, manifest)
        manifest["manifest_path"] = path
        return manifest

    input_uri = gcs_input_uri(bucket, suite, target, arm, stage)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_prefix = gcs_output_prefix(bucket, suite, target, arm, stage).rstrip("/") + f"/{stamp}/"
    gcs_io.upload_file(local_jsonl, input_uri, project=proj)

    client = make_vertex_client(project=proj, location=loc)
    from google.genai.types import CreateBatchJobConfig

    display = f"mr-eval-{suite}-{target}-{arm}-{stage}"
    job = client.batches.create(
        model=model,
        src=input_uri,
        config=CreateBatchJobConfig(dest=output_prefix, display_name=display),
    )
    manifest = {
        "suite": suite,
        "target": target,
        "arm": arm,
        "stage": stage,
        "model": model,
        "project": proj,
        "location": loc,
        "bucket": bucket,
        "input_uri": input_uri,
        "output_uri_prefix": output_prefix,
        "local_request_jsonl": local_jsonl,
        "job_name": getattr(job, "name", None),
        "job_state": str(getattr(job, "state", None)),
        "display_name": display,
        "n_requests": n_lines,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "skipped_empty": False,
    }
    path = local_manifest(suite, target, arm, stage)
    _write_manifest(path, manifest)
    manifest["manifest_path"] = path
    return manifest


def load_manifest(suite: str, target: str, arm: str, stage: str) -> Dict[str, Any]:
    path = local_manifest(suite, target, arm, stage)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No manifest at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def refresh_stage(
    *,
    suite: str,
    target: str,
    arm: str,
    stage: str,
    project: Optional[str] = None,
    location: Optional[str] = None,
) -> Dict[str, Any]:
    manifest = load_manifest(suite, target, arm, stage)
    if manifest.get("skipped_empty"):
        return {
            "suite": suite,
            "target": target,
            "arm": arm,
            "stage": stage,
            "state": "SKIPPED_EMPTY",
            "job_name": None,
            "error": None,
        }
    job_name = manifest.get("job_name")
    if not job_name:
        raise RuntimeError(f"Manifest missing job_name for {suite}/{target}/{arm}/{stage}")

    proj = get_project(project or manifest.get("project"))
    loc = get_location(location or manifest.get("location") or "global")
    client = make_vertex_client(project=proj, location=loc)
    job = client.batches.get(name=job_name)
    state = getattr(job, "state", None)
    state_name = getattr(state, "name", None) or str(state)
    err = str(getattr(job, "error", None) or "") or None
    manifest["job_state"] = state_name
    if err:
        manifest["job_error"] = err
    _write_manifest(local_manifest(suite, target, arm, stage), manifest)
    return {
        "suite": suite,
        "target": target,
        "arm": arm,
        "stage": stage,
        "state": state_name,
        "job_name": job_name,
        "error": err,
    }
