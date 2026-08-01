"""Poll Vertex Batch job status."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from management_reasoning.batch.submit import load_manifest
from management_reasoning.clients.vertex import get_location, get_project, make_vertex_client


def refresh_job(
    *,
    provider: str,
    arm: str,
    suite: str = "primary",
    project: Optional[str] = None,
    location: Optional[str] = None,
) -> Dict[str, Any]:
    manifest = load_manifest(provider, arm, suite=suite)
    job_name = manifest.get("job_name")
    if not job_name:
        raise RuntimeError(f"Manifest missing job_name for {provider}/{arm}")

    proj = get_project(project or manifest.get("project"))
    # Prefer location from manifest (Claude batch uses us-east5, not global).
    loc = get_location(location or manifest.get("location"))
    client = make_vertex_client(project=proj, location=loc)
    job = client.batches.get(name=job_name)

    state = getattr(job, "state", None)
    state_s = str(state) if state is not None else None
    # Prefer enum .name when present
    state_name = getattr(state, "name", None) or state_s

    out = {
        "provider": provider,
        "arm": arm,
        "suite": suite,
        "job_name": job_name,
        "state": state_name,
        "error": str(getattr(job, "error", None) or "") or None,
        "output_uri_prefix": manifest.get("output_uri_prefix"),
        "dest": getattr(job, "dest", None) or getattr(job, "output_uri", None),
    }

    # Persist refreshed state onto manifest
    from management_reasoning.batch.paths import local_manifest_path
    import os

    manifest["job_state"] = state_name
    if out["error"]:
        manifest["job_error"] = out["error"]
    path = local_manifest_path(provider, arm, suite=suite)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return out
