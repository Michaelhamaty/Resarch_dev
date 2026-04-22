"""Deterministic stub adapter used for Phase 2 scaffolding and tests.

The stub does no real inference. It emits a fixed shape of markdown with
an HTML ``<table>`` block so Phase 3's verifier can still exercise
structural checks against the output, and it derives everything
deterministically from ``(page_id, model_name, budget_name, prompt_id)``
so re-running a smoke job produces byte-identical artifacts. Real model
integration plugs in via ``factory.build_adapter`` by adding a new
``adapter_kind``.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from PIL import Image

from ..config.budgets import Budget
from ..config.prompts import PromptTemplate
from .adapter import InferenceAdapter
from .types import InferenceResult


class StubInferenceAdapter(InferenceAdapter):
    """Returns a deterministic fake page parse. No model is loaded."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def run(
        self,
        page_id: str,
        image: Image.Image,
        budget: Budget,
        prompt: PromptTemplate,
    ) -> InferenceResult:
        started_at = _iso_now()
        t0 = time.perf_counter()

        # Touch the image so we exercise the loader contract without
        # needing a real model. Width/height are embedded in the stub
        # output for traceability during debugging.
        width, height = image.size

        raw_text = _render_stub_markdown(
            page_id=page_id,
            model_name=self.model_name,
            budget_name=budget.name,
            prompt_id=prompt.id,
            image_wh=(width, height),
        )

        runtime_ms = (time.perf_counter() - t0) * 1000.0
        finished_at = _iso_now()

        return InferenceResult(
            page_id=page_id,
            model_name=self.model_name,
            budget_name=budget.name,
            prompt_id=prompt.id,
            raw_text=raw_text,
            output_token_count=_rough_token_count(raw_text),
            runtime_ms=runtime_ms,
            started_at=started_at,
            finished_at=finished_at,
        )


def _render_stub_markdown(
    *,
    page_id: str,
    model_name: str,
    budget_name: str,
    prompt_id: str,
    image_wh: tuple[int, int],
) -> str:
    """Build a small, deterministic page with one HTML table block.

    The body is prose-free aside from a single heading line. This matches
    the prompt contract (no explanatory prose) and gives the Phase 3
    verifier a real ``<table>`` to parse.
    """

    w, h = image_wh
    return (
        f"# {page_id}\n"
        "\n"
        "<table>\n"
        "  <tr><th>column_a</th><th>column_b</th></tr>\n"
        f"  <tr><td>{page_id}</td><td>{model_name}</td></tr>\n"
        f"  <tr><td>{budget_name}</td><td>{prompt_id}</td></tr>\n"
        f"  <tr><td>image_w</td><td>{w}</td></tr>\n"
        f"  <tr><td>image_h</td><td>{h}</td></tr>\n"
        "</table>\n"
    )


def _rough_token_count(text: str) -> int:
    """Whitespace-split token count. Good enough for Phase 2 log fields."""

    return len(text.split())


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()
