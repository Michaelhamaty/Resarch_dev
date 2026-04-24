"""Pick the frozen adaptive pair and matched-cost fixed baselines.

Two selection functions:

- :func:`select_adaptive_pair` — adaptive ``(B_low, B_high)`` closest
  to the configured ``target_adaptive_cost_tiles``. Tie-break: smallest
  ``B_low``, then smallest ``B_high``.
- :func:`select_matched_fixed` — fixed budget within a relative
  tolerance of the adaptive pair's *measured* cost. Tie-break: smallest
  ``|cost - target|``, then smallest ``max_tiles``. Records whether
  the pick fell outside the tolerance band as a documented fallback.

Both functions are pure: no I/O, no side effects. The caller is
responsible for passing in sweep-point lists (see ``sweep.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

from .sweep import AdaptiveSweepPoint, FixedSweepPoint


@dataclass(frozen=True)
class SelectedAdaptive:
    low_max_tiles: int
    high_max_tiles: int
    measured_cost_tiles: float
    target_cost_tiles: float
    point: AdaptiveSweepPoint


@dataclass(frozen=True)
class SelectedFixed:
    max_tiles: int
    measured_cost_tiles: float
    target_cost_tiles: float
    within_tolerance: bool
    point: FixedSweepPoint


def select_adaptive_pair(
    points: list[AdaptiveSweepPoint], target_cost_tiles: float
) -> SelectedAdaptive:
    if not points:
        raise ValueError("select_adaptive_pair requires at least one sweep point")

    def sort_key(p: AdaptiveSweepPoint) -> tuple[float, int, int]:
        return (
            abs(p.summary.cost_tiles - target_cost_tiles),
            p.low_max_tiles,
            p.high_max_tiles,
        )

    winner = min(points, key=sort_key)
    return SelectedAdaptive(
        low_max_tiles=winner.low_max_tiles,
        high_max_tiles=winner.high_max_tiles,
        measured_cost_tiles=winner.summary.cost_tiles,
        target_cost_tiles=target_cost_tiles,
        point=winner,
    )


def select_matched_fixed(
    points: list[FixedSweepPoint],
    target_cost_tiles: float,
    tolerance: float,
) -> SelectedFixed:
    """Select the fixed candidate whose cost best matches the adaptive target.

    Preference order:
    1. Candidates with relative deviation ``<= tolerance``: pick by
       smallest ``|cost - target|``, then smallest ``max_tiles``.
    2. If no candidate is within tolerance, fall back to the
       absolute-closest (same tie-break) and mark ``within_tolerance=False``
       in the result so the artifact can record the fallback.
    """

    if not points:
        raise ValueError("select_matched_fixed requires at least one sweep point")
    if target_cost_tiles <= 0:
        raise ValueError(
            f"target_cost_tiles must be positive, got {target_cost_tiles}"
        )

    def deviation(p: FixedSweepPoint) -> float:
        return abs(p.summary.cost_tiles - target_cost_tiles) / target_cost_tiles

    def sort_key(p: FixedSweepPoint) -> tuple[float, int]:
        return (
            abs(p.summary.cost_tiles - target_cost_tiles),
            p.max_tiles,
        )

    in_band = [p for p in points if deviation(p) <= tolerance]
    pool = in_band if in_band else points
    winner = min(pool, key=sort_key)

    return SelectedFixed(
        max_tiles=winner.max_tiles,
        measured_cost_tiles=winner.summary.cost_tiles,
        target_cost_tiles=target_cost_tiles,
        within_tolerance=bool(in_band),
        point=winner,
    )
