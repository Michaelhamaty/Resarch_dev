"""Cheap HTML-table metadata helpers shared across dataset loaders.

Both the OmniDocBench and FinTabNet loaders need the same approximate
row-count / col-count / merged-cell / nested-header signals to decide
whether a page is "complex" enough for HardTableRule and for stratified
sampling in scale-up v2. The regex-based parsing here is intentionally
cheap; full HTML parsing happens later in the verifier and scorer where
correctness matters more than speed.
"""

from __future__ import annotations

import re

_TR_RE = re.compile(r"<tr\b[^>]*>", re.IGNORECASE)
_TD_RE = re.compile(r"<t[dh]\b[^>]*>", re.IGNORECASE)
_CELL_WITH_BODY_RE = re.compile(
    r"<(t[dh])\b[^>]*>(.*?)</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")
_SPAN_GT1_RE = re.compile(r"\b(?:rowspan|colspan)\s*=\s*['\"]?(\d+)", re.IGNORECASE)
_THEAD_RE = re.compile(r"<thead\b[^>]*>", re.IGNORECASE)
_TR_END_RE = re.compile(r"</tr\s*>", re.IGNORECASE)
_TH_RE = re.compile(r"<th\b[^>]*>", re.IGNORECASE)


def count_non_empty_cells(html: str) -> int:
    """Count ``<td>``/``<th>`` cells whose body has non-whitespace text."""

    count = 0
    for match in _CELL_WITH_BODY_RE.finditer(html):
        body = _TAG_RE.sub("", match.group(2))
        if body.strip():
            count += 1
    return count


def table_metadata(html: str) -> tuple[int, int, bool, bool]:
    """Return ``(row_count, col_count, has_merged_cells, has_nested_headers)``."""

    rows = _TR_RE.findall(html)
    row_count = len(rows)

    col_count = 0
    for row_segment in _split_rows(html):
        cells = len(_TD_RE.findall(row_segment))
        if cells > col_count:
            col_count = cells

    has_merged = any(int(m.group(1)) > 1 for m in _SPAN_GT1_RE.finditer(html))
    has_nested = _has_nested_headers(html)
    return row_count, col_count, has_merged, has_nested


def _split_rows(html: str) -> list[str]:
    rows: list[str] = []
    for opener in _TR_RE.finditer(html):
        end_match = _TR_END_RE.search(html, opener.end())
        if not end_match:
            break
        rows.append(html[opener.end() : end_match.start()])
    return rows


def _has_nested_headers(html: str) -> bool:
    if len(_THEAD_RE.findall(html)) > 1:
        return True
    for row_segment in _split_rows(html):
        if len(_TH_RE.findall(row_segment)) > 1:
            return True
    return False
