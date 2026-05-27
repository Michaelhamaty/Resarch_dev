"""Tests for paired Wilcoxon signed-rank wrapper."""

from __future__ import annotations

import pytest

from adaptive_inference.analysis.paired_tests import (
    align_pages,
    paired_wilcoxon,
)


def test_identical_arrays_return_p1():
    a = [0.5, 0.6, 0.7, 0.8]
    res = paired_wilcoxon(a, list(a))
    assert res.p_value == pytest.approx(1.0)
    assert res.n_pairs == 4
    assert res.n_dropped_ties == 4


def test_strictly_better_b_returns_small_p():
    a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    b = [x + 0.1 for x in a]
    res = paired_wilcoxon(a, b)
    # b > a on every pair → strong effect → small p.
    assert res.p_value < 0.05


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        paired_wilcoxon([0.1], [0.1, 0.2])


def test_empty_raises():
    with pytest.raises(ValueError):
        paired_wilcoxon([], [])


def test_ties_dropped_but_remaining_pairs_tested():
    a = [0.5, 0.5, 0.5, 0.1]
    b = [0.5, 0.5, 0.5, 0.4]
    res = paired_wilcoxon(a, b)
    assert res.n_pairs == 4
    assert res.n_dropped_ties == 3
    # Only one nonzero diff; p-value is permissive but defined.
    assert 0.0 <= res.p_value <= 1.0


def test_align_pages_intersects_and_sorts():
    a = {"x": 0.1, "y": 0.2, "z": 0.3}
    b = {"y": 0.5, "z": 0.6, "w": 0.7}
    ids, va, vb = align_pages(a, b)
    assert ids == ["y", "z"]
    assert va == [0.2, 0.3]
    assert vb == [0.5, 0.6]


def test_align_pages_empty_intersection():
    ids, va, vb = align_pages({"a": 1}, {"b": 2})
    assert ids == [] and va == [] and vb == []
