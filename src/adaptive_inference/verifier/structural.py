"""Top-level entry point: run all structural checks over a page's raw output.

This module composes the Phase 3 pipeline — extraction, span expansion,
rectangular consistency, and degenerate-table detection — into a single
``verify_page_tables(raw_text)`` call that returns a ``VerifierResult``.
It is deliberately pure: no disk I/O, no logging, no policy decisions
beyond the precision-first PASS/REPARSE rule. Phase 4's runner wires it
into the page state machine; this module stays decoupled from that.
"""

from __future__ import annotations

from ..parsing.html_tables import extract_table_blocks
from .codes import (
    DECISION_PASS,
    DECISION_REPARSE,
    DEGENERATE_TABLE,
    HTML_PARSE_ERROR,
    NO_TABLE_FOUND,
    RECTANGULAR_INCONSISTENCY,
    SPAN_EXPANSION_FAILED,
)
from .spans import (
    HtmlParseError,
    RectangularInconsistencyError,
    SpanExpansionError,
    expand_to_grid,
)
from .types import TableSummary, VerifierResult


def verify_page_tables(raw_text: str) -> VerifierResult:
    """Inspect raw page output and return a deterministic verifier decision."""

    blocks = extract_table_blocks(raw_text)

    if not blocks:
        return VerifierResult(
            decision=DECISION_REPARSE,
            failure_codes=(NO_TABLE_FOUND,),
            predicted_table_count=0,
            html_parse_ok=True,
            span_normalization_ok=True,
            tables=(),
        )

    summaries = tuple(_check_table(i, block) for i, block in enumerate(blocks))

    html_parse_ok = all(HTML_PARSE_ERROR not in s.failure_codes for s in summaries)
    span_normalization_ok = all(
        SPAN_EXPANSION_FAILED not in s.failure_codes
        and RECTANGULAR_INCONSISTENCY not in s.failure_codes
        for s in summaries
    )

    page_codes = _page_codes(summaries)
    decision = DECISION_PASS if not page_codes else DECISION_REPARSE

    return VerifierResult(
        decision=decision,
        failure_codes=page_codes,
        predicted_table_count=len(summaries),
        html_parse_ok=html_parse_ok,
        span_normalization_ok=span_normalization_ok,
        tables=summaries,
    )


def _check_table(index: int, table_html: str) -> TableSummary:
    try:
        grid = expand_to_grid(table_html)
    except HtmlParseError:
        return TableSummary(index=index, rows=0, cols=0, failure_codes=(HTML_PARSE_ERROR,))
    except SpanExpansionError:
        return TableSummary(
            index=index, rows=0, cols=0, failure_codes=(SPAN_EXPANSION_FAILED,)
        )
    except RectangularInconsistencyError:
        return TableSummary(
            index=index, rows=0, cols=0, failure_codes=(RECTANGULAR_INCONSISTENCY,)
        )

    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    if rows == 0 or _all_cells_empty(grid):
        return TableSummary(
            index=index, rows=rows, cols=cols, failure_codes=(DEGENERATE_TABLE,)
        )

    return TableSummary(index=index, rows=rows, cols=cols, failure_codes=())


def _all_cells_empty(grid: tuple[tuple[str, ...], ...]) -> bool:
    for row in grid:
        for cell in row:
            if cell:
                return False
    return True


def _page_codes(summaries: tuple[TableSummary, ...]) -> tuple[str, ...]:
    """Stable-order union of per-table failure codes. No duplicates."""

    seen: list[str] = []
    for summary in summaries:
        for code in summary.failure_codes:
            if code not in seen:
                seen.append(code)
    return tuple(seen)
