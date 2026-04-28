"""Cost-summary layer for Phase 7.

Cost is in ``max_tiles`` — the same proxy Phase 5 used to pick budgets.
We delegate every per-page formula to ``calibration.cost`` so there is
exactly one cost-accounting source of truth in the repo.

For each system we report:

- the frozen budget(s) consumed,
- the *target* cost (from the frozen artifact's selection block),
- the *measured* mean per-page cost on the held-out split,
- the absolute and relative delta.

For the four canonical adaptive comparisons (adaptive_2b vs.
fixed_2b_low / fixed_2b_matched / fixed_8b_matched), we additionally
report the cost-axis-only spread so the analysis report can show
"matched-budget" as a single fact rather than three separate numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..calibration.artifact import FrozenBudgets
from ..calibration.cost import adaptive_cost_tiles, fixed_cost_tiles
from .loaders import LoadedSystem
from .results import ADAPTIVE_RUNNERS, SINGLE_PASS_RUNNER


@dataclass(frozen=True)
class SystemCost:
    """One row per system in the cost summary."""

    system_id: str
    family: str
    runner: str
    status: str
    measured_cost_tiles: float
    target_cost_tiles: float | None
    delta_tiles: float | None
    delta_relative: float | None
    budget_label: str


@dataclass(frozen=True)
class CostSummary:
    """Top-level cost summary written to disk by Phase 7."""

    cost_unit: str
    target_adaptive_cost_tiles: float
    matched_cost_tolerance: float
    systems: tuple[SystemCost, ...]


def build_cost_summary(
    systems: tuple[LoadedSystem, ...], frozen: FrozenBudgets
) -> CostSummary:
    """Compute measured per-system cost and tolerance deltas."""

    target_adaptive = frozen.adaptive_selection.target_cost_tiles
    rows: list[SystemCost] = []
    for sys in systems:
        rows.append(_one(sys, frozen, target_adaptive))
    return CostSummary(
        cost_unit="max_tiles",
        target_adaptive_cost_tiles=target_adaptive,
        matched_cost_tolerance=frozen.matched_cost_tolerance,
        systems=tuple(rows),
    )


def _one(
    sys: LoadedSystem,
    frozen: FrozenBudgets,
    target_adaptive: float,
) -> SystemCost:
    entry = sys.entry
    if entry.status != "ok":
        return SystemCost(
            system_id=entry.system_id,
            family=entry.family,
            runner=entry.runner,
            status=entry.status,
            measured_cost_tiles=0.0,
            target_cost_tiles=None,
            delta_tiles=None,
            delta_relative=None,
            budget_label=_budget_label(entry),
        )

    if entry.runner in ADAPTIVE_RUNNERS:
        if entry.budget_low_max_tiles is None or entry.budget_high_max_tiles is None:
            raise ValueError(
                f"{entry.system_id}: adaptive entry must carry low+high tile budgets"
            )
        measured = adaptive_cost_tiles(
            sys.records,
            entry.budget_low_max_tiles,
            entry.budget_high_max_tiles,
        )
        target = target_adaptive
    elif entry.runner == SINGLE_PASS_RUNNER:
        if entry.budget_max_tiles is None:
            raise ValueError(
                f"{entry.system_id}: single_pass entry must carry budget_max_tiles"
            )
        measured = fixed_cost_tiles(entry.budget_max_tiles)
        target = _fixed_target(entry, frozen)
    else:
        raise ValueError(f"{entry.system_id}: unknown runner {entry.runner!r}")

    delta_tiles = measured - target if target is not None else None
    delta_relative = (
        delta_tiles / target if (target is not None and target > 0 and delta_tiles is not None) else None
    )
    return SystemCost(
        system_id=entry.system_id,
        family=entry.family,
        runner=entry.runner,
        status=entry.status,
        measured_cost_tiles=measured,
        target_cost_tiles=target,
        delta_tiles=delta_tiles,
        delta_relative=delta_relative,
        budget_label=_budget_label(entry),
    )


def _fixed_target(entry, frozen: FrozenBudgets) -> float | None:
    """Map a fixed system's budget name back to its frozen selection target."""

    if entry.budget_name == frozen.b_fix_2b.name:
        return frozen.fixed_2b_selection.target_cost_tiles
    if entry.budget_name == frozen.b_fix_8b.name:
        return frozen.fixed_8b_selection.target_cost_tiles
    if entry.budget_name == frozen.b_low.name:
        # fixed_2b_low has no target in the frozen artifact (it's the floor
        # baseline). Reporting None keeps that explicit.
        return None
    return None


def _budget_label(entry) -> str:
    if entry.budget_name and entry.budget_max_tiles is not None:
        return f"{entry.budget_name}={entry.budget_max_tiles}"
    if (
        entry.budget_low_name
        and entry.budget_low_max_tiles is not None
        and entry.budget_high_name
        and entry.budget_high_max_tiles is not None
    ):
        return (
            f"{entry.budget_low_name}={entry.budget_low_max_tiles}"
            f" / {entry.budget_high_name}={entry.budget_high_max_tiles}"
        )
    return "(no budget)"


def to_dict(summary: CostSummary) -> Mapping[str, object]:
    """Stable dict shape for JSON output."""

    return {
        "cost_unit": summary.cost_unit,
        "target_adaptive_cost_tiles": summary.target_adaptive_cost_tiles,
        "matched_cost_tolerance": summary.matched_cost_tolerance,
        "systems": [
            {
                "system_id": s.system_id,
                "family": s.family,
                "runner": s.runner,
                "status": s.status,
                "budget_label": s.budget_label,
                "measured_cost_tiles": s.measured_cost_tiles,
                "target_cost_tiles": s.target_cost_tiles,
                "delta_tiles": s.delta_tiles,
                "delta_relative": s.delta_relative,
            }
            for s in summary.systems
        ],
    }
