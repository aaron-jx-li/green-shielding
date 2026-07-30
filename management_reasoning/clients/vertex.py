"""Shared Vertex AI client helpers (Application Default Credentials)."""

from __future__ import annotations

import os
import subprocess
from typing import Any, Optional

from management_reasoning.config import models_config

_SETUP_HINT = (
    "Vertex client requires org GCP setup:\n"
    "  1) Enable Vertex AI on the org project\n"
    "  2) gcloud auth application-default login\n"
    "  3) export GOOGLE_CLOUD_PROJECT=...  (or: gcloud config set project ...)\n"
    "  4) optionally export GOOGLE_CLOUD_LOCATION=global  "
    "(needed for gemini-3-pro-preview and other global-only models)\n"
    "See management_reasoning/README.md (GCP setup checklist)."
)


class VertexConfigError(RuntimeError):
    """Missing project / ADC configuration for Vertex."""


def _gcloud_project() -> Optional[str]:
    try:
        proc = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    val = (proc.stdout or "").strip()
    if not val or val == "(unset)":
        return None
    return val


def get_project(project: Optional[str] = None) -> str:
    """
    Resolve GCP project id.

    Order: explicit arg → GOOGLE_CLOUD_PROJECT / GCLOUD_PROJECT →
    gcloud config → models.yaml project_default → error.
    """
    if project and str(project).strip():
        return str(project).strip()
    env = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")
    if env and str(env).strip():
        return str(env).strip()
    from_gcloud = _gcloud_project()
    if from_gcloud:
        return from_gcloud
    cfg = models_config().get("project_default", "").strip()
    if cfg:
        return cfg
    raise VertexConfigError("Could not resolve GCP project.\n" + _SETUP_HINT)


def get_location(location: Optional[str] = None) -> str:
    if location and str(location).strip():
        return str(location).strip()
    env = os.environ.get("GOOGLE_CLOUD_LOCATION")
    if env and str(env).strip():
        return str(env).strip()
    cfg = models_config().get("location_default", "").strip()
    return cfg or "global"


def make_vertex_client(
    *,
    project: Optional[str] = None,
    location: Optional[str] = None,
) -> Any:
    """Build a google-genai Client bound to Vertex AI (not API-key Gemini)."""
    proj = get_project(project)
    loc = get_location(location)
    try:
        from google import genai
    except ImportError as e:
        raise VertexConfigError(
            "google-genai is not installed. pip install google-genai\n" + _SETUP_HINT
        ) from e

    try:
        return genai.Client(vertexai=True, project=proj, location=loc)
    except Exception as e:
        raise VertexConfigError(
            f"Failed to create Vertex client (project={proj}, location={loc}): {e}\n"
            + _SETUP_HINT
        ) from e
