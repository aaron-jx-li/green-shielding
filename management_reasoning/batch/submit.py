"""Submit Vertex Batch jobs for management_reasoning."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from management_reasoning.batch.paths import (
    batch_model_resource,
    gcs_input_uri,
    gcs_output_prefix,
    local_manifest_path,
    local_request_jsonl,
)
from management_reasoning.batch import gcs_io
from management_reasoning.clients.vertex import get_location, get_project, make_vertex_client


def _write_manifest(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def submit_job(
    *,
    provider: str,
    arm: str,
    model: str,
    bucket: str,
    suite: str = "primary",
    project: Optional[str] = None,
    location: Optional[str] = None,
    display_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Upload local requests.jsonl (if present) and create a BatchPredictionJob.

    Expects prepare to have written ``local_request_jsonl``.
    Claude Batch is not supported on the ``global`` endpoint; default to ``us-east5``.
    """
    proj = get_project(project)
    if location:
        loc = get_location(location)
    elif provider == "claude":
        # Vertex Claude Batch rejects global (Gemini-only).
        loc = get_location("us-east5")
    else:
        loc = get_location(None)
    local_jsonl = local_request_jsonl(provider, arm, suite=suite)
    if not os.path.isfile(local_jsonl):
        raise FileNotFoundError(
            f"Missing {local_jsonl}. Run prepare first for {provider}/{arm}."
        )

    input_uri = gcs_input_uri(bucket, provider, arm, suite=suite)
    # Unique output prefix per submit so re-runs don't mix prediction folders.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_prefix = gcs_output_prefix(bucket, provider, arm, suite=suite).rstrip("/") + f"/{stamp}/"
    gcs_io.upload_file(local_jsonl, input_uri, project=proj)

    client = make_vertex_client(project=proj, location=loc)
    model_resource = batch_model_resource(provider, model)
    job_display = display_name or f"mgmt-{suite}-{provider}-{arm}"

    from google.genai.types import CreateBatchJobConfig

    job = client.batches.create(
        model=model_resource,
        src=input_uri,
        config=CreateBatchJobConfig(
            dest=output_prefix,
            display_name=job_display,
        ),
    )

    manifest = {
        "suite": suite,
        "provider": provider,
        "arm": arm,
        "model": model,
        "model_resource": model_resource,
        "project": proj,
        "location": loc,
        "bucket": bucket,
        "input_uri": input_uri,
        "output_uri_prefix": output_prefix,
        "local_request_jsonl": local_jsonl,
        "job_name": getattr(job, "name", None),
        "job_state": str(getattr(job, "state", None)),
        "display_name": job_display,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = local_manifest_path(provider, arm, suite=suite)
    _write_manifest(manifest_path, manifest)
    manifest["manifest_path"] = manifest_path
    return manifest


def load_manifest(provider: str, arm: str, *, suite: str = "primary") -> Dict[str, Any]:
    path = local_manifest_path(provider, arm, suite=suite)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No manifest at {path}. Submit first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
