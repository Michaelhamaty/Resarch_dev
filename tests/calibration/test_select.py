"""Unit tests for selection logic (adaptive pair + matched fixed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_inference.calibration.select import (
    select_adaptive_pair,
    select_matched_fixed,
)
from adaptive_inference.calibration.summary import SweepPointSummary
from adaptive_inference.calibration.sweep import AdaptiveSweepPoint, FixedSweepPoint


def _adaptive_point(low: int, high: int, cost_tiles: float) -> AdaptiveSweepPoint:
    return AdaptiveSweepPoint(
        low_max_tiles=low,
        high_max_tiles=high,
        summary=SweepPointSummary(
            sample_size=5,
            mean_runtime_ms=1.0,
            p95_runtime_ms=1.0,
            mean_output_tokens=10.0,
            mean_tile_budget=cost_tiles,
            reparse_rate=0.0,
            cost_tiles=cost_tiles,
        ),
        output_dir=Path("/tmp/does-not-matter"),
    )


def _fixed_point(max_tiles: int, cost_tiles: float) -> FixedSweepPoint:
    return FixedSweepPoint(
        model_name="m",
        max_tiles=max_tiles,
        summary=SweepPointSummary(
            sample_size=5,
            mean_runtime_ms=1.0,
            p95_runtime_ms=1.0,
            mean_output_tokens=10.0,
            mean_tile_budget=cost_tiles,
            reparse_rate=None,
            cost_tiles=cost_tiles,
        ),
        output_dir=Path("/tmp/does-not-matter"),
    )


# ---------- adaptive pair selection --------------------------------------


def test_select_adaptive_picks_closest_to_target() -> None:
    points = [
        _adaptive_point(low=2, high=12, cost_tiles=2.0),
        _adaptive_point(low=4, high=12, cost_tiles=4.0),
        _adaptive_point(low=6, high=16, cost_tiles=6.0),
    ]
    sel = select_adaptive_pair(points, target_cost_tiles=5.0)
    # 4.0 and 6.0 are equidistant from 5.0; tie-break → smallest low.
    assert sel.low_max_tiles == 4
    assert sel.high_max_tiles == 12
    assert sel.measured_cost_tiles == 4.0


def test_select_adaptive_ties_prefer_smaller_high() -> None:
    # All three have identical cost; tie-break: smallest low, then smallest high.
    points = [
        _adaptive_point(low=4, high=16, cost_tiles=4.0),
        _adaptive_point(low=4, high=8, cost_tiles=4.0),
        _adaptive_point(low=4, high=12, cost_tiles=4.0),
    ]
    sel = select_adaptive_pair(points, target_cost_tiles=4.0)
    assert sel.low_max_tiles == 4
    assert sel.high_max_tiles == 8


def test_select_adaptive_empty_raises() -> None:
    with pytest.raises(ValueError):
        select_adaptive_pair([], target_cost_tiles=5.0)


# ---------- matched fixed selection --------------------------------------


def test_select_matched_fixed_within_tolerance() -> None:
    points = [
        _fixed_point(4, cost_tiles=4.0),
        _fixed_point(6, cost_tiles=6.0),
        _fixed_point(8, cost_tiles=8.0),
        _fixed_point(12, cost_tiles=12.0),
    ]
    sel = select_matched_fixed(points, target_cost_tiles=6.1, tolerance=0.10)
    assert sel.max_tiles == 6
    assert sel.measured_cost_tiles == 6.0
    assert sel.within_tolerance is True


def test_select_matched_fixed_ties_prefer_smallest_max_tiles() -> None:
    # Two equidistant candidates, both within tolerance: pick smaller max_tiles.
    points = [
        _fixed_point(4, cost_tiles=4.0),
        _fixed_point(8, cost_tiles=8.0),
    ]
    sel = select_matched_fixed(points, target_cost_tiles=6.0, tolerance=0.50)
    assert sel.max_tiles == 4
    assert sel.within_tolerance is True


def test_select_matched_fixed_fallback_when_all_outside_tolerance() -> None:
    # Target 6.0 with tolerance 0.05 → only deviations <= 0.30 allowed.
    # None of {4.0, 8.0, 12.0} qualify; fall back to absolute-closest.
    points = [
        _fixed_point(4, cost_tiles=4.0),  # deviation 2.0
        _fixed_point(8, cost_tiles=8.0),  # deviation 2.0
        _fixed_point(12, cost_tiles=12.0),  # deviation 6.0
    ]
    sel = select_matched_fixed(points, target_cost_tiles=6.0, tolerance=0.05)
    # Tie between 4 and 8 by |cost - target|; tie-break to smaller max_tiles.
    assert sel.max_tiles == 4
    assert sel.within_tolerance is False


def test_select_matched_fixed_empty_raises() -> None:
    with pytest.raises(ValueError):
        select_matched_fixed([], target_cost_tiles=6.0, tolerance=0.10)


def test_select_matched_fixed_rejects_non_positive_target() -> None:
    with pytest.raises(ValueError):
        select_matched_fixed(
            [_fixed_point(4, cost_tiles=4.0)],
            target_cost_tiles=0.0,
            tolerance=0.10,
        )
