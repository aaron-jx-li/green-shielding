"""
Multi-stage Vertex Batch for management_reasoning diagnosis eval judges.

Judge model: ``gemini-3.1-flash-lite`` (thinking HIGH). Targets scored: Gemini/Claude
× raw/neutralized primary responses.

CLI: ``python -m management_reasoning.eval.batch {prepare,submit,status,collect,aggregate}``

See ``scripts/management_reasoning/smoke_eval_batch.sh`` for the n=3 smoke flow.
"""
