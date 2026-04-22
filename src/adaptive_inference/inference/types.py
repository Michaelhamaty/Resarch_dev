"""Shared inference result dataclass.

``InferenceResult`` is the one object every adapter returns — stub today,
real InternVL2 later. Keeping this shape stable means the runner, output
writer, and runtime logger never need to care which adapter produced a
result.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceResult:
    page_id: str
    model_name: str
    budget_name: str
    prompt_id: str
    raw_text: str
    output_token_count: int
    runtime_ms: float
    started_at: str  # ISO-8601 UTC, e.g. "2026-04-21T12:34:56.789012+00:00"
    finished_at: str
