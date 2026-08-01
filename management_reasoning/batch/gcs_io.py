"""GCS upload/download helpers for management_reasoning batch."""

from __future__ import annotations

import os
from typing import List, Optional
from urllib.parse import urlparse


def _parse_gs(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    parsed = urlparse(uri)
    bucket = parsed.netloc
    path = parsed.path.lstrip("/")
    return bucket, path


def _client(project: Optional[str] = None):
    try:
        from google.cloud import storage
    except ImportError as e:
        raise RuntimeError(
            "google-cloud-storage is required for batch GCS I/O. "
            "pip install google-cloud-storage"
        ) from e
    return storage.Client(project=project) if project else storage.Client()


def upload_file(local_path: str, gcs_uri: str, *, project: Optional[str] = None) -> str:
    bucket_name, blob_name = _parse_gs(gcs_uri)
    client = _client(project)
    blob = client.bucket(bucket_name).blob(blob_name)
    blob.upload_from_filename(local_path)
    return gcs_uri


def download_file(gcs_uri: str, local_path: str, *, project: Optional[str] = None) -> str:
    bucket_name, blob_name = _parse_gs(gcs_uri)
    client = _client(project)
    blob = client.bucket(bucket_name).blob(blob_name)
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    blob.download_to_filename(local_path)
    return local_path


def list_prefix(gcs_prefix: str, *, project: Optional[str] = None) -> List[str]:
    """List gs:// blob URIs under a prefix."""
    prefix_uri = gcs_prefix if gcs_prefix.endswith("/") else gcs_prefix + "/"
    bucket_name, prefix = _parse_gs(prefix_uri)
    client = _client(project)
    uris: List[str] = []
    for blob in client.list_blobs(bucket_name, prefix=prefix):
        if blob.name.endswith("/"):
            continue
        uris.append(f"gs://{bucket_name}/{blob.name}")
    return uris


def download_prefix(
    gcs_prefix: str,
    local_dir: str,
    *,
    project: Optional[str] = None,
    suffix: str = ".jsonl",
) -> List[str]:
    """Download matching blobs under prefix into local_dir; preserve relative paths."""
    prefix_uri = gcs_prefix if gcs_prefix.endswith("/") else gcs_prefix + "/"
    bucket_name, prefix = _parse_gs(prefix_uri)
    os.makedirs(local_dir, exist_ok=True)
    local_paths: List[str] = []
    for uri in list_prefix(gcs_prefix, project=project):
        if suffix and not uri.endswith(suffix):
            continue
        _, blob_name = _parse_gs(uri)
        rel = blob_name[len(prefix) :] if blob_name.startswith(prefix) else os.path.basename(blob_name)
        local_path = os.path.join(local_dir, rel)
        download_file(uri, local_path, project=project)
        local_paths.append(local_path)
    return local_paths
