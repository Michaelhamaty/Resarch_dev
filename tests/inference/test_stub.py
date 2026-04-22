"""Tests for the deterministic stub inference adapter."""

from __future__ import annotations

from PIL import Image

from adaptive_inference.config.budgets import Budget
from adaptive_inference.config.prompts import PromptTemplate
from adaptive_inference.inference.factory import build_adapter
from adaptive_inference.config.models import ModelConfig
from adaptive_inference.inference.stub import StubInferenceAdapter


BUDGET = Budget(name="low", max_tiles=4)
PROMPT = PromptTemplate(id="table_parse_v1", version=1, description="d", template="t")


def _img() -> Image.Image:
    return Image.new("L", (8, 8), color=42)


def test_stub_result_has_required_fields() -> None:
    adapter = StubInferenceAdapter(model_name="internvl2-2b")
    r = adapter.run(page_id="page_0001", image=_img(), budget=BUDGET, prompt=PROMPT)
    assert r.page_id == "page_0001"
    assert r.model_name == "internvl2-2b"
    assert r.budget_name == "low"
    assert r.prompt_id == "table_parse_v1"
    assert r.output_token_count > 0
    assert r.runtime_ms >= 0.0
    assert r.started_at and r.finished_at


def test_stub_raw_text_contains_html_table() -> None:
    adapter = StubInferenceAdapter(model_name="internvl2-2b")
    r = adapter.run(page_id="page_0001", image=_img(), budget=BUDGET, prompt=PROMPT)
    assert "<table>" in r.raw_text
    assert "</table>" in r.raw_text
    assert "<tr>" in r.raw_text


def test_stub_raw_text_is_deterministic() -> None:
    adapter = StubInferenceAdapter(model_name="internvl2-2b")
    a = adapter.run(page_id="page_0001", image=_img(), budget=BUDGET, prompt=PROMPT)
    b = adapter.run(page_id="page_0001", image=_img(), budget=BUDGET, prompt=PROMPT)
    assert a.raw_text == b.raw_text


def test_stub_raw_text_varies_by_page() -> None:
    adapter = StubInferenceAdapter(model_name="internvl2-2b")
    a = adapter.run(page_id="page_0001", image=_img(), budget=BUDGET, prompt=PROMPT)
    b = adapter.run(page_id="page_0002", image=_img(), budget=BUDGET, prompt=PROMPT)
    assert a.raw_text != b.raw_text


def test_factory_dispatches_stub() -> None:
    cfg = ModelConfig(
        name="internvl2-2b",
        adapter_kind="stub",
        model_id="x",
        notes="",
    )
    adapter = build_adapter(cfg)
    assert isinstance(adapter, StubInferenceAdapter)
    assert adapter.model_name == "internvl2-2b"


def test_factory_rejects_unknown_kind() -> None:
    cfg = ModelConfig(
        name="m",
        adapter_kind="not-a-thing",
        model_id="x",
        notes="",
    )
    try:
        build_adapter(cfg)
    except ValueError as e:
        assert "Unknown adapter_kind" in str(e)
    else:  # pragma: no cover - failure if it did not raise
        raise AssertionError("expected ValueError")
