from __future__ import annotations

import pytest

from adaptive_inference.verifier.spans import (
    HtmlParseError,
    RectangularInconsistencyError,
    SpanExpansionError,
    expand_to_grid,
)


def _table(body: str) -> str:
    return f"<table>{body}</table>"


def test_plain_2x2_grid():
    grid = expand_to_grid(
        _table("<tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr>")
    )
    assert grid == (("a", "b"), ("c", "d"))


def test_colspan_expansion_fills_row():
    grid = expand_to_grid(
        _table('<tr><td colspan="2">ab</td></tr><tr><td>c</td><td>d</td></tr>')
    )
    assert grid == (("ab", ""), ("c", "d"))


def test_rowspan_expansion_fills_column():
    grid = expand_to_grid(
        _table(
            '<tr><td rowspan="2">A</td><td>b</td></tr>'
            "<tr><td>c</td></tr>"
        )
    )
    assert grid == (("A", "b"), ("", "c"))


def test_combined_rowspan_colspan():
    grid = expand_to_grid(
        _table(
            '<tr><td rowspan="2" colspan="2">big</td><td>x</td></tr>'
            "<tr><td>y</td></tr>"
        )
    )
    assert grid == (("big", "", "x"), ("", "", "y"))


def test_empty_table_returns_empty_grid():
    assert expand_to_grid("<table></table>") == ()


def test_table_with_empty_row_returns_empty_grid():
    assert expand_to_grid("<table><tr></tr></table>") == ()


def test_th_cells_are_treated_as_cells():
    grid = expand_to_grid(
        _table("<tr><th>h1</th><th>h2</th></tr><tr><td>a</td><td>b</td></tr>")
    )
    assert grid == (("h1", "h2"), ("a", "b"))


def test_zero_colspan_is_rejected():
    with pytest.raises(SpanExpansionError):
        expand_to_grid(_table('<tr><td colspan="0">x</td></tr>'))


def test_negative_rowspan_is_rejected():
    with pytest.raises(SpanExpansionError):
        expand_to_grid(_table('<tr><td rowspan="-1">x</td></tr>'))


def test_non_integer_colspan_is_rejected():
    with pytest.raises(SpanExpansionError):
        expand_to_grid(_table('<tr><td colspan="two">x</td></tr>'))


def test_overlapping_rowspan_is_rejected():
    # Row 0's second cell has rowspan=3 at column 2. Row 1's second
    # cell is a colspan=3 whose extent would cover column 2, which is
    # already occupied by the descending span -> overlap error.
    body = (
        '<tr><td colspan="2">A</td><td rowspan="3">B</td></tr>'
        '<tr><td>C</td><td colspan="3">D</td></tr>'
    )
    with pytest.raises(SpanExpansionError):
        expand_to_grid(_table(body))


def test_ragged_rows_raise_rectangular_inconsistency():
    body = "<tr><td>a</td><td>b</td><td>c</td></tr><tr><td>d</td><td>e</td></tr>"
    with pytest.raises(RectangularInconsistencyError):
        expand_to_grid(_table(body))


def test_empty_string_raises_parse_error():
    with pytest.raises(HtmlParseError):
        expand_to_grid("")


def test_non_table_html_raises_parse_error():
    with pytest.raises(HtmlParseError):
        expand_to_grid("<div>not a table</div>")


def test_nested_tables_are_not_recursed():
    # Outer table has one <tr><td> containing an inner <table>. Only the
    # outer table's rows should drive expansion; the inner table's rows
    # are ignored (nested tables are out of MVP scope).
    body = (
        "<tr><td>outer_cell"
        "<table><tr><td>inner_a</td><td>inner_b</td></tr></table>"
        "</td></tr>"
    )
    grid = expand_to_grid(_table(body))
    assert len(grid) == 1
    assert len(grid[0]) == 1
    assert "outer_cell" in grid[0][0]
