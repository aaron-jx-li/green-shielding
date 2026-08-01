"""
Diagnosis-side evaluation for management_reasoning target outputs.

Separate from ``open_eval/`` (OpenAI judges). Uses Vertex Gemini Flash-Lite
with thinking. Pure metric aggregators are imported read-only from
``open_eval.eval.metrics`` to stay aligned with the paper radar axes.
"""
