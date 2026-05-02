"""Unit tests for the cell-F1 + text-similarity scorer.

These tests exercise the scorer in isolation: no runner, no adapter, no
data on disk. The intent is to prove the metric behaves predictably on
hand-crafted inputs before we point it at real model output.
"""

from __future__ import annotations

import pytest

from adaptive_inference.scoring import ScoreResult, score_page


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _table(rows: list[list[str]]) -> str:
    """Build a minimal HTML table from a 2D list of cell texts."""

    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table>{body}</table>"


GOLD_2X2 = _table([["a", "b"], ["c", "d"]])


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


def test_identical_tables_score_perfect() -> None:
    res = score_page(GOLD_2X2, GOLD_2X2)
    assert res.cell_precision == 1.0
    assert res.cell_recall == 1.0
    assert res.cell_f1 == 1.0
    assert res.text_similarity == 1.0
    assert res.pred_parse_error is None
    assert res.gold_cell_count == 4
    assert res.pred_cell_count == 4


def test_completely_different_text_zero_f1() -> None:
    pred = _table([["x", "y"], ["z", "w"]])
    res = score_page(pred, GOLD_2X2)
    assert res.cell_f1 == 0.0
    # Same shape, different content: text_similarity is positive (separators match).
    assert 0.0 < res.text_similarity < 1.0


def test_pred_missing_row_drops_recall_keeps_precision() -> None:
    pred = _table([["a", "b"]])  # missing the second row
    res = score_page(pred, GOLD_2X2)
    assert res.cell_precision == 1.0
    assert res.cell_recall == pytest.approx(0.5)
    assert res.cell_f1 == pytest.approx(2 / 3)


def test_pred_extra_row_drops_precision_keeps_recall() -> None:
    pred = _table([["a", "b"], ["c", "d"], ["e", "f"]])
    res = score_page(pred, GOLD_2X2)
    assert res.cell_precision == pytest.approx(4 / 6)
    assert res.cell_recall == 1.0


def test_span_pred_matches_unspanned_gold_when_grid_equal() -> None:
    """A merged cell expands to repeated positions; both grids must match cell-for-cell."""

    gold = _table([["X", "X"], ["c", "d"]])
    pred = "<table><tr><td colspan='2'>X</td></tr><tr><td>c</td><td>d</td></tr></table>"
    res = score_page(pred, gold)
    assert res.cell_f1 == 1.0


def test_malformed_pred_records_parse_error_and_zero_f1() -> None:
    pred = "<table><tr><td>oops"  # never closed
    res = score_page(pred, GOLD_2X2)
    assert res.pred_parse_error is not None
    assert res.cell_f1 == 0.0
    assert res.pred_cell_count == 0


def test_no_tables_in_pred_records_no_tables_error() -> None:
    res = score_page("just some prose, no html here", GOLD_2X2)
    assert res.pred_parse_error == "no_tables"
    assert res.cell_f1 == 0.0


def test_ragged_pred_records_not_rectangular_error() -> None:
    pred = "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td></tr></table>"
    res = score_page(pred, GOLD_2X2)
    assert res.pred_parse_error == "not_rectangular"
    assert res.cell_f1 == 0.0


def test_whitespace_collapse_in_cell_text() -> None:
    pred = _table([["a  b", "c\nd"], ["e", "f"]])
    gold = _table([["a b", "c d"], ["e", "f"]])
    res = score_page(pred, gold)
    assert res.cell_f1 == 1.0


def test_case_is_preserved() -> None:
    pred = _table([["A", "b"], ["c", "d"]])
    res = score_page(pred, GOLD_2X2)
    # 3/4 cells match; 'A' != 'a' under our normalization.
    assert res.cell_f1 == pytest.approx(0.75)


def test_multiple_tables_aligned_by_position() -> None:
    pred = GOLD_2X2 + GOLD_2X2  # two identical tables
    gold = GOLD_2X2 + GOLD_2X2
    res = score_page(pred, gold)
    assert res.cell_f1 == 1.0
    assert res.gold_cell_count == 8


def test_multiple_tables_different_count_partial_credit() -> None:
    pred = GOLD_2X2  # one table
    gold = GOLD_2X2 + GOLD_2X2  # two tables
    res = score_page(pred, gold)
    assert res.cell_recall == pytest.approx(0.5)
    assert res.cell_precision == 1.0


def test_empty_inputs_score_one() -> None:
    """Both sides agree there is nothing to predict — convention says perfect."""

    res = score_page("<table></table>", "<table></table>")
    assert res.cell_f1 == 1.0
    assert res.text_similarity == 1.0
    assert res.gold_cell_count == 0
    assert res.pred_cell_count == 0


def test_score_result_is_immutable() -> None:
    res = score_page(GOLD_2X2, GOLD_2X2)
    assert isinstance(res, ScoreResult)
    with pytest.raises(Exception):
        res.cell_f1 = 0.0  # type: ignore[misc]


def test_bad_gold_html_raises() -> None:
    with pytest.raises(ValueError):
        score_page(GOLD_2X2, "no tables in gold either")


def test_text_similarity_in_unit_interval() -> None:
    pred = _table([["aaa", "b"], ["c", "d"]])
    gold = _table([["aab", "b"], ["c", "d"]])
    res = score_page(pred, gold)
    assert 0.0 <= res.text_similarity <= 1.0
    assert res.text_similarity > 0.8  # one-character substitution out of ~9
