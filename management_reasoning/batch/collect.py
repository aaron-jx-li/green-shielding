"""Download Vertex Batch outputs and write local responses.jsonl."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from management_reasoning.batch import gcs_io
from management_reasoning.batch.paths import (
    is_legacy_freeform_suite,
    local_manifest_dir,
    local_responses_dir,
    parse_custom_id,
)
from management_reasoning.batch.submit import load_manifest
from management_reasoning.schema import parse_model_response, parse_single_question_response


def _extract_text_from_gemini_response(resp: Any) -> str:
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp.strip()
    if not isinstance(resp, dict):
        return str(resp).strip()

    # Direct text
    if isinstance(resp.get("text"), str):
        return resp["text"].strip()

    # candidates[0].content.parts[].text
    candidates = resp.get("candidates") or []
    parts_text: List[str] = []
    for cand in candidates:
        content = (cand or {}).get("content") or {}
        for part in content.get("parts") or []:
            t = (part or {}).get("text")
            if t:
                parts_text.append(t)
    if parts_text:
        return "".join(parts_text).strip()

    # nested response
    if "response" in resp:
        return _extract_text_from_gemini_response(resp["response"])
    return ""


def _extract_text_from_claude_response(resp: Any) -> str:
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp.strip()
    if not isinstance(resp, dict):
        return str(resp).strip()

    # Anthropic message content blocks
    content = resp.get("content")
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
            elif isinstance(block, dict) and "text" in block:
                parts.append(block.get("text") or "")
            elif isinstance(block, str):
                parts.append(block)
        if parts:
            return "".join(parts).strip()
    if isinstance(content, str):
        return content.strip()

    # Sometimes wrapped
    for key in ("message", "response", "body"):
        if key in resp:
            text = _extract_text_from_claude_response(resp[key])
            if text:
                return text
    return ""


def _extract_usage(provider: str, obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    usage = obj.get("usage")
    if usage is None and isinstance(obj.get("response"), dict):
        usage = obj["response"].get("usage")
    if usage is None and isinstance(obj.get("response"), dict):
        usage = obj["response"].get("usageMetadata") or obj["response"].get("usage_metadata")

    # Gemini usageMetadata on response
    resp = obj.get("response")
    if provider == "gemini" and isinstance(resp, dict):
        um = resp.get("usageMetadata") or resp.get("usage_metadata")
        if isinstance(um, dict):
            out = {}
            for k in (
                "promptTokenCount",
                "candidatesTokenCount",
                "thoughtsTokenCount",
                "totalTokenCount",
                "prompt_token_count",
                "candidates_token_count",
                "thoughts_token_count",
                "total_token_count",
            ):
                if k in um and um[k] is not None:
                    snake = (
                        k.replace("TokenCount", "_token_count")
                        .replace("promptToken", "prompt_token")
                        .replace("candidatesToken", "candidates_token")
                        .replace("thoughtsToken", "thoughts_token")
                        .replace("totalToken", "total_token")
                    )
                    if "Token" in k:
                        # camelCase → snake
                        mapping = {
                            "promptTokenCount": "prompt_token_count",
                            "candidatesTokenCount": "candidates_token_count",
                            "thoughtsTokenCount": "thoughts_token_count",
                            "totalTokenCount": "total_token_count",
                        }
                        out[mapping.get(k, k)] = int(um[k])
                    else:
                        out[k] = int(um[k])
            cand = out.get("candidates_token_count") or 0
            thoughts = out.get("thoughts_token_count") or 0
            out["billed_output_token_count"] = cand + thoughts
            return out or None

    if isinstance(usage, dict):
        out = dict(usage)
        if provider == "claude":
            if "input_tokens" in out:
                out["prompt_token_count"] = out["input_tokens"]
            if "output_tokens" in out:
                out["billed_output_token_count"] = out["output_tokens"]
                out["candidates_token_count"] = out["output_tokens"]
        return out
    return None


def _raw_from_line(provider: str, obj: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """Return (raw_text, error)."""
    status = obj.get("status")
    if status and status not in ("", "OK", "ok", {}):
        if isinstance(status, dict) and status.get("code") not in (None, 0, "OK", "ok"):
            return "", json.dumps(status, ensure_ascii=False)
        if isinstance(status, str) and status.upper() not in ("OK", "SUCCEEDED", ""):
            # some outputs use status as error string
            if "error" in status.lower() or "fail" in status.lower():
                return "", status

    resp = obj.get("response")
    if resp is None:
        # sometimes the prediction is the whole object
        resp = obj.get("prediction") or obj

    if provider == "gemini":
        text = _extract_text_from_gemini_response(resp)
    else:
        text = _extract_text_from_claude_response(resp)

    if not text:
        err = obj.get("error")
        if err:
            return "", json.dumps(err, ensure_ascii=False) if not isinstance(err, str) else err
        return "", "empty model response in batch output"
    return text, None


def iter_batch_output_rows(paths: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def _select_prediction_files(paths: List[str]) -> List[str]:
    """
    Prefer final ``predictions.jsonl`` files; skip incremental chunks.

    When multiple job folders share an output prefix (re-submits), keep only the
    newest ``prediction-model-*/predictions.jsonl`` by directory name.
    """
    finals = [
        p
        for p in paths
        if os.path.basename(p) == "predictions.jsonl"
        and "incremental_predictions" not in p.replace("\\", "/")
    ]
    if not finals:
        # Fall back to any non-empty jsonl
        return [p for p in paths if os.path.getsize(p) > 0]

    # Group by parent prediction-model-* directory timestamp in path
    def sort_key(p: str) -> str:
        parts = p.replace("\\", "/").split("/")
        for part in parts:
            if part.startswith("prediction-model-"):
                return part
        return p

    finals_sorted = sorted(finals, key=sort_key)
    newest = finals_sorted[-1]
    # If several prediction-model dirs, only the newest; if flat, keep all finals
    newest_dir_token = sort_key(newest)
    if newest_dir_token.startswith("prediction-model-"):
        return [p for p in finals_sorted if sort_key(p) == newest_dir_token]
    return finals_sorted


def collect_job(
    *,
    provider: str,
    arm: str,
    suite: str = "primary",
    tag: str = "primary_batch",
    project: Optional[str] = None,
) -> Dict[str, Any]:
    manifest = load_manifest(provider, arm, suite=suite)
    model = manifest["model"]
    proj = project or manifest.get("project")
    output_prefix = manifest["output_uri_prefix"]
    job_name = manifest.get("job_name")

    dl_dir = os.path.join(local_manifest_dir(provider, arm, suite=suite), "output_raw")
    # Clear previous downloads so re-collects don't mix old flat files
    if os.path.isdir(dl_dir):
        import shutil

        shutil.rmtree(dl_dir)
    local_files = gcs_io.download_prefix(output_prefix, dl_dir, project=proj, suffix=".jsonl")
    local_files = _select_prediction_files(local_files)
    if not local_files:
        raise RuntimeError(
            f"No .jsonl outputs under {output_prefix}. "
            f"Is the job finished? Listed files: {gcs_io.list_prefix(output_prefix, project=proj)}"
        )

    out_dir = local_responses_dir(model, arm, tag=tag, suite=suite)
    out_jsonl = os.path.join(out_dir, "responses.jsonl")
    # Safety: never let a non-primary suite write into *_primary_batch by mistake.
    if suite != "primary" and tag == "primary_batch":
        raise RuntimeError(
            f"Refusing to collect suite={suite!r} into tag={tag!r} "
            f"(would overwrite primary). Use tag={suite}_batch or omit --tag."
        )
    if suite == "primary" and tag != "primary_batch" and "primary" not in tag:
        # allow explicit overrides, but warn via stderr
        print(
            f"WARNING: collecting primary suite into non-default tag={tag!r}",
            file=__import__("sys").stderr,
        )
    os.makedirs(out_dir, exist_ok=True)

    freeform = is_legacy_freeform_suite(suite)

    stats = {
        "n": 0,
        "parse_ok": 0,
        "errors": 0,
        "missing_custom_id": 0,
        "duplicates_skipped": 0,
    }
    seen: set[str] = set()
    with open(out_jsonl, "w", encoding="utf-8") as out_f:
        for obj in iter_batch_output_rows(local_files):
            cid = obj.get("custom_id") or obj.get("customId")
            if not cid:
                stats["missing_custom_id"] += 1
                continue
            cid_s = str(cid)
            if cid_s in seen:
                stats["duplicates_skipped"] += 1
                continue
            try:
                parts = parse_custom_id(cid_s)
            except ValueError:
                stats["missing_custom_id"] += 1
                continue
            if parts.provider != provider or parts.arm != arm:
                continue

            raw, gen_err = _raw_from_line(provider, obj)
            record: Dict[str, Any] = {
                "sample_id": parts.sample_id,
                "arm": arm,
                "provider": provider,
                "model": model,
                "project": manifest.get("project"),
                "location": manifest.get("location"),
                "raw_response": raw or None,
                "model_response": raw or None,
                "parsed": None,
                "parse_ok": False,
                "error": gen_err,
                "refusal": False,
                "usage": _extract_usage(provider, obj),
                "custom_id": cid_s,
                "batch_job_name": job_name,
                "order_id": parts.order_id,
                "question_id": parts.question_id,
                "suite": suite,
            }
            if raw:
                if freeform:
                    # Paper free-form: success = non-empty prose (no JSON schema).
                    record["parse_ok"] = True
                    record["parsed"] = None
                elif parts.question_id:
                    parsed = parse_single_question_response(raw, parts.question_id)
                    record["parse_ok"] = parsed.parse_ok
                    record["parsed"] = parsed.parsed
                    record["refusal"] = parsed.refusal
                    if parsed.error and not record["error"]:
                        record["error"] = parsed.error
                else:
                    parsed = parse_model_response(raw)
                    record["parse_ok"] = parsed.parse_ok
                    record["parsed"] = parsed.parsed
                    record["refusal"] = parsed.refusal
                    if parsed.error and not record["error"]:
                        record["error"] = parsed.error
            seen.add(cid_s)
            stats["n"] += 1
            if record["parse_ok"]:
                stats["parse_ok"] += 1
            if record.get("error") and not record["parse_ok"]:
                stats["errors"] += 1
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "provider": provider,
        "arm": arm,
        "out_jsonl": out_jsonl,
        "downloaded_files": local_files,
        **stats,
    }
