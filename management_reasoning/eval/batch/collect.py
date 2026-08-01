"""Collect judge Batch outputs → per-stage collected.jsonl."""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, List, Optional

from management_reasoning.batch import gcs_io
from management_reasoning.batch.collect import (
    _extract_text_from_gemini_response,
    _select_prediction_files,
    iter_batch_output_rows,
)
from management_reasoning.eval.batch.paths import local_collected, local_manifest, local_root
from management_reasoning.eval.batch.submit import load_manifest
from management_reasoning.eval.json_utils import robust_json_loads
from management_reasoning.eval.batch.paths import parse_custom_id


def collect_stage(
    *,
    suite: str,
    target: str,
    arm: str,
    stage: str,
    project: Optional[str] = None,
) -> Dict[str, Any]:
    manifest = load_manifest(suite, target, arm, stage)
    if manifest.get("skipped_empty"):
        out = local_collected(suite, target, arm, stage)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        open(out, "w").close()
        return {"n": 0, "parse_ok": 0, "errors": 0, "out_jsonl": out, "skipped_empty": True}

    proj = project or manifest.get("project")
    output_prefix = manifest["output_uri_prefix"]
    dl_dir = os.path.join(local_root(suite, target, arm, stage), "output_raw")
    if os.path.isdir(dl_dir):
        shutil.rmtree(dl_dir)
    local_files = gcs_io.download_prefix(output_prefix, dl_dir, project=proj, suffix=".jsonl")
    local_files = _select_prediction_files(local_files)
    if not local_files:
        raise RuntimeError(f"No predictions under {output_prefix}")

    rows_in = iter_batch_output_rows(local_files)
    out_path = local_collected(suite, target, arm, stage)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    n = 0
    parse_ok = 0
    errors = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for obj in rows_in:
            n += 1
            cid = obj.get("custom_id") or ""
            try:
                parts = parse_custom_id(cid)
            except Exception as e:
                errors += 1
                f.write(
                    json.dumps(
                        {"custom_id": cid, "parse_ok": False, "error": f"bad_id:{e}"},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            resp = obj.get("response") or obj.get("prediction") or obj
            text = _extract_text_from_gemini_response(resp)
            rec: Dict[str, Any] = {
                "custom_id": cid,
                "sample_id": parts["sample_id"],
                "stage": parts["stage"],
                "target": parts["target"],
                "arm": parts["arm"],
                "chunk": parts.get("chunk"),
                "raw_response": text,
                "parsed": None,
                "parse_ok": False,
                "error": None,
            }
            if not text:
                rec["error"] = "empty response"
                errors += 1
            else:
                try:
                    parsed = robust_json_loads(text)
                    rec["parsed"] = parsed
                    rec["parse_ok"] = True
                    # Flatten common fields for later stages
                    if stage == "extract":
                        rec["extracted_diagnoses"] = parsed.get("extracted_diagnoses") or []
                        rec["top_k_diagnoses"] = parsed.get("top_k_diagnoses") or []
                    elif stage == "unc":
                        rec["uncertainty_flag"] = bool(parsed.get("uncertainty_flag", False))
                    elif stage == "sem":
                        rec["matches"] = parsed.get("matches") or []
                    elif stage == "ground":
                        rec["per_diagnosis"] = parsed.get("per_diagnosis") or []
                    parse_ok += 1
                except Exception as e:
                    rec["error"] = f"json:{e}"
                    errors += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {"n": n, "parse_ok": parse_ok, "errors": errors, "out_jsonl": out_path}
