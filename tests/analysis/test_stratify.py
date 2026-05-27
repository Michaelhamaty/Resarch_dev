"""Tests for difficulty-bucket stratification of per-page scores."""

from __future__ import annotations

from adaptive_inference.analysis.stratify import stratify_by_difficulty


def _meta(row_count: int, merged: bool = False):
    return {"row_count": row_count, "has_merged_cells": merged}


def test_buckets_by_row_count():
    scores = {"a": 0.5, "b": 0.6, "c": 0.7}
    meta = {"a": _meta(3), "b": _meta(10), "c": _meta(20)}
    out = stratify_by_difficulty(scores, meta)
    assert [p for p, _ in out["simple"]] == ["a"]
    assert [p for p, _ in out["complex"]] == ["b"]
    assert [p for p, _ in out["very_complex"]] == ["c"]


def test_merged_promotes_to_very_complex():
    scores = {"a": 0.4}
    meta = {"a": _meta(3, merged=True)}
    out = stratify_by_difficulty(scores, meta)
    assert out["simple"] == []
    assert [p for p, _ in out["very_complex"]] == ["a"]


def test_unbucketable_page_dropped():
    scores = {"a": 0.4, "ghost": 0.9}
    meta = {"a": _meta(3)}
    out = stratify_by_difficulty(scores, meta)
    assert all("ghost" not in [p for p, _ in v] for v in out.values())


def test_within_bucket_sorted_by_page_id():
    scores = {"z": 0.1, "a": 0.2, "m": 0.3}
    meta = {pid: _meta(3) for pid in scores}
    out = stratify_by_difficulty(scores, meta)
    assert [p for p, _ in out["simple"]] == ["a", "m", "z"]
