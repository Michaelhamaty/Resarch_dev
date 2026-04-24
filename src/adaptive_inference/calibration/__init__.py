"""Phase 5 calibration: sweep and freeze the four budgets used by Phase 6.

The calibration layer is a composition on top of the Phase 2 and Phase 4
runners. It does not change any of their behavior; it only drives them
over candidate budgets on the frozen calibration split and distills the
results into a single, auditable artifact.

Public surface:

- :class:`CalibrationConfig` / :func:`load_calibration_config`
- :class:`SweepPointSummary`
- :func:`sweep_adaptive` / :func:`sweep_fixed`
- :func:`select_adaptive_pair` / :func:`select_matched_fixed`
- :class:`FrozenBudgets` / :func:`write_frozen_budgets` / :func:`load_frozen_budgets`
"""

from .artifact import (
    FrozenBudget,
    FrozenBudgets,
    load_frozen_budgets,
    write_frozen_budgets,
)
from .config import CalibrationConfig, load_calibration_config
from .cost import adaptive_cost_tiles, fixed_cost_tiles
from .select import (
    SelectedAdaptive,
    SelectedFixed,
    select_adaptive_pair,
    select_matched_fixed,
)
from .summary import SweepPointSummary, summarize_adaptive_log, summarize_single_pass_log
from .sweep import AdaptiveSweepPoint, FixedSweepPoint, sweep_adaptive, sweep_fixed

__all__ = [
    "AdaptiveSweepPoint",
    "CalibrationConfig",
    "FixedSweepPoint",
    "FrozenBudget",
    "FrozenBudgets",
    "SelectedAdaptive",
    "SelectedFixed",
    "SweepPointSummary",
    "adaptive_cost_tiles",
    "fixed_cost_tiles",
    "load_calibration_config",
    "load_frozen_budgets",
    "select_adaptive_pair",
    "select_matched_fixed",
    "summarize_adaptive_log",
    "summarize_single_pass_log",
    "sweep_adaptive",
    "sweep_fixed",
    "write_frozen_budgets",
]
