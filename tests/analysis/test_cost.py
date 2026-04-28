"""Cost-summary tests.

Verifies Phase 7 cost reuses the Phase 5 cost helpers (single source of
truth) and produces the right deltas relative to the frozen target.
"""

from __future__ import annotations

import pytest

from adaptive_inference.analysis import cost as analysis_cost
from adaptive_inference.analysis.cost import build_cost_summary, to_dict
from adaptive_inference.analysis.loaders import (
    load_loaded_systems,
    load_phase6_manifest,
)
from adaptive_inference.calibration.artifact import load_frozen_budgets
from adaptive_inference.calibration.cost import (
    adaptive_cost_tiles,
    fixed_cost_tiles,
)


def test_cost_module_reuses_calibration_helpers():
    """Phase 7 cost wires through the Phase 5 cost helpers, not its own copies."""

    src = analysis_cost.__dict__
    assert src["adaptive_cost_tiles"] is adaptive_cost_tiles
    assert src["fixed_cost_tiles"] is fixed_cost_tiles


def test_build_cost_summary_reports_targets_and_deltas(phase7_fixture):
    manifest = load_phase6_manifest(phase7_fixture.phase6_manifest_path)
    systems = load_loaded_systems(manifest, repo_root=phase7_fixture.root)
    frozen = load_frozen_budgets(phase7_fixture.frozen_budgets_path)

    summary = build_cost_summary(systems, frozen)
    assert summary.cost_unit == "max_tiles"
    assert summary.target_adaptive_cost_tiles == pytest.approx(6.0)

    by_id = {s.system_id: s for s in summary.systems}

    adaptive = by_id["adaptive_2b"]
    assert adaptive.measured_cost_tiles == pytest.approx(6.0)
    assert adaptive.target_cost_tiles == pytest.approx(6.0)
    assert adaptive.delta_tiles == pytest.approx(0.0)
    assert adaptive.delta_relative == pytest.approx(0.0)
    assert "B_low" in adaptive.budget_label and "B_high" in adaptive.budget_label

    fix2 = by_id["fixed_2b_matched"]
    assert fix2.measured_cost_tiles == pytest.approx(6.0)
    assert fix2.target_cost_tiles == pytest.approx(6.0)
    assert fix2.delta_tiles == pytest.approx(0.0)

    # fixed_2b_low has no target (it's the floor baseline)
    fix_low = by_id["fixed_2b_low"]
    assert fix_low.target_cost_tiles is None
    assert fix_low.delta_tiles is None


def test_to_dict_round_trip(phase7_fixture):
    manifest = load_phase6_manifest(phase7_fixture.phase6_manifest_path)
    systems = load_loaded_systems(manifest, repo_root=phase7_fixture.root)
    frozen = load_frozen_budgets(phase7_fixture.frozen_budgets_path)
    payload = to_dict(build_cost_summary(systems, frozen))
    assert payload["cost_unit"] == "max_tiles"
    assert isinstance(payload["systems"], list)
    assert len(payload["systems"]) == 6
