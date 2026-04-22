"""Shape/immutability tests for the shared ``InferenceResult`` dataclass."""

from __future__ import annotations

import pytest

from adaptive_inference.inference.types import InferenceResult


def _make() -> InferenceResult:
    return InferenceResult(
        page_id="page_0001",
        model_name="internvl2-2b",
        budget_name="low",
        prompt_id="table_parse_v1",
        raw_text="hello",
        output_token_count=1,
        runtime_ms=0.1,
        started_at="2026-04-21T00:00:00+00:00",
        finished_at="2026-04-21T00:00:01+00:00",
    )


def test_inference_result_fields() -> None:
    r = _make()
    assert r.page_id == "page_0001"
    assert r.output_token_count == 1
    assert r.budget_name == "low"


def test_inference_result_is_frozen() -> None:
    r = _make()
    with pytest.raises(Exception):
        r.page_id = "other"  # type: ignore[misc]
