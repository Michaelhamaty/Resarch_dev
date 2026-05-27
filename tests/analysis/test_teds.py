"""Tests for the TEDS scorer."""

from __future__ import annotations

import pytest

from adaptive_inference.analysis.teds import teds_score


_SIMPLE = """
<table>
  <thead><tr><th>A</th><th>B</th></tr></thead>
  <tbody>
    <tr><td>1</td><td>2</td></tr>
    <tr><td>3</td><td>4</td></tr>
  </tbody>
</table>
"""


def test_identical_html_returns_1():
    assert teds_score(_SIMPLE, _SIMPLE) == pytest.approx(1.0)


def test_whitespace_does_not_change_score():
    spaced = _SIMPLE.replace(">", ">  ").replace("<", "  <")
    assert teds_score(spaced, _SIMPLE) == pytest.approx(1.0)


def test_one_cell_text_changed_drops_below_1():
    perturbed = _SIMPLE.replace(">1<", ">9<")
    score = teds_score(perturbed, _SIMPLE)
    assert 0.0 < score < 1.0


def test_completely_different_structure_low_score():
    other = "<table><tr><td>X</td></tr></table>"
    score = teds_score(other, _SIMPLE)
    assert score < 0.5


def test_unparseable_pred_returns_zero():
    assert teds_score("no table here", _SIMPLE) == 0.0


def test_unparseable_gold_returns_zero():
    assert teds_score(_SIMPLE, "") == 0.0


def test_missing_table_returns_zero():
    assert teds_score("<div><p>no</p></div>", _SIMPLE) == 0.0


def test_extra_row_drops_score_but_remains_positive():
    extra = _SIMPLE.replace("</tbody>", "<tr><td>5</td><td>6</td></tr></tbody>")
    score = teds_score(extra, _SIMPLE)
    assert 0.0 < score < 1.0


def test_swapped_cell_texts_within_same_row_lowers_score():
    swapped = _SIMPLE.replace(">A<", ">TMP<").replace(">B<", ">A<").replace(">TMP<", ">B<")
    score = teds_score(swapped, _SIMPLE)
    assert score < 1.0
