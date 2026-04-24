"""Unit tests for the per-page cost formulas."""

from __future__ import annotations

import pytest

from adaptive_inference.calibration.cost import adaptive_cost_tiles, fixed_cost_tiles


def test_fixed_cost_is_max_tiles() -> None:
    assert fixed_cost_tiles(4) == 4.0
    assert fixed_cost_tiles(12) == 12.0


def test_fixed_cost_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        fixed_cost_tiles(0)
    with pytest.raises(ValueError):
        fixed_cost_tiles(-3)


def test_adaptive_cost_no_reparse() -> None:
    records = [{"reparse_triggered": False}, {"reparse_triggered": False}]
    assert adaptive_cost_tiles(records, low_max_tiles=4, high_max_tiles=12) == 4.0


def test_adaptive_cost_all_reparse() -> None:
    records = [{"reparse_triggered": True}, {"reparse_triggered": True}]
    # Both passes run: 4 + 12 = 16 per page.
    assert adaptive_cost_tiles(records, low_max_tiles=4, high_max_tiles=12) == 16.0


def test_adaptive_cost_mixed_is_weighted_mean() -> None:
    records = [
        {"reparse_triggered": True},
        {"reparse_triggered": False},
        {"reparse_triggered": False},
        {"reparse_triggered": False},
    ]
    # reparse_rate = 0.25. cost = 4 + 0.25*12 = 7.0
    assert adaptive_cost_tiles(records, low_max_tiles=4, high_max_tiles=12) == 7.0


def test_adaptive_cost_empty_returns_zero() -> None:
    assert adaptive_cost_tiles([], low_max_tiles=4, high_max_tiles=12) == 0.0


def test_adaptive_cost_rejects_non_positive_budgets() -> None:
    with pytest.raises(ValueError):
        adaptive_cost_tiles(
            [{"reparse_triggered": False}], low_max_tiles=0, high_max_tiles=12
        )
    with pytest.raises(ValueError):
        adaptive_cost_tiles(
            [{"reparse_triggered": False}], low_max_tiles=4, high_max_tiles=-1
        )
