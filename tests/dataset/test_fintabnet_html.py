"""Tests for the apoidea/fintabnet-html HTML-shape selector."""

from __future__ import annotations

import pytest

from adaptive_inference.dataset.fintabnet_html import (
    HtmlRow,
    select_html_pages,
)


_SIMPLE = (
    "<table>"
    "<tr><td>a</td><td>b</td></tr>"
    "<tr><td>1</td><td>2</td></tr>"
    "</table>"
)
_COMPLEX = (
    "<table><thead>"
    "<tr><th>h1</th><th>h2</th><th>h3</th></tr>"
    "</thead><tbody>"
    "<tr><td colspan=\"2\">merged</td><td>x</td></tr>"
    "<tr><td>a</td><td>b</td><td>c</td></tr>"
    "</tbody></table>"
)


def _row(html: str, split: str = "validation") -> HtmlRow:
    return HtmlRow(html_table=html, image_bytes=b"\x89PNG", split=split)


def test_selects_validation_split_only() -> None:
    rows = [
        _row(_SIMPLE, split="validation"),
        _row(_COMPLEX, split="train"),
    ]
    out = select_html_pages(rows, limit=10)
    assert len(out) == 1
    assert out[0].table_html == _SIMPLE


def test_caller_can_open_train_split_explicitly() -> None:
    rows = [_row(_SIMPLE, split="validation"), _row(_COMPLEX, split="train")]
    out = select_html_pages(rows, limit=10, splits=("validation", "train"))
    assert len(out) == 2


def test_dropps_empty_html() -> None:
    rows = [_row(""), _row("   \n  "), _row(_SIMPLE)]
    out = select_html_pages(rows, limit=10)
    assert len(out) == 1


def test_min_non_empty_cells_filter() -> None:
    sparse = "<table><tr><td>x</td><td></td></tr></table>"
    rows = [_row(sparse), _row(_SIMPLE)]
    out = select_html_pages(rows, limit=10, min_non_empty_cells=3)
    assert len(out) == 1
    assert out[0].table_html == _SIMPLE


def test_metadata_populated_from_html() -> None:
    out = select_html_pages([_row(_COMPLEX)], limit=10)
    assert len(out) == 1
    page = out[0]
    assert page.row_count == 3
    assert page.col_count == 3
    assert page.has_merged_cells is True
    assert page.has_nested_headers is True
    assert page.language == "en"


def test_page_id_is_content_addressed_and_stable() -> None:
    out1 = select_html_pages([_row(_SIMPLE)], limit=10)
    out2 = select_html_pages([_row(_SIMPLE)], limit=10)
    assert out1[0].page_id == out2[0].page_id
    assert out1[0].page_id.startswith("fintabnet_")
    assert out1[0].image_filename == out1[0].page_id + ".png"


def test_duplicate_html_dedupes_across_rows() -> None:
    out = select_html_pages([_row(_SIMPLE), _row(_SIMPLE)], limit=10)
    assert len(out) == 1


def test_output_sorted_by_page_id() -> None:
    out = select_html_pages([_row(_COMPLEX), _row(_SIMPLE)], limit=10)
    assert [p.page_id for p in out] == sorted(p.page_id for p in out)


def test_limit_applied_after_sort() -> None:
    out = select_html_pages([_row(_COMPLEX), _row(_SIMPLE)], limit=1)
    assert len(out) == 1


def test_invalid_limit_rejected() -> None:
    with pytest.raises(ValueError):
        select_html_pages([_row(_SIMPLE)], limit=0)


def test_invalid_min_cells_rejected() -> None:
    with pytest.raises(ValueError):
        select_html_pages([_row(_SIMPLE)], limit=1, min_non_empty_cells=-1)
