"""Unit tests for Phase 6 frozen-inputs translation layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_inference.calibration.artifact import (
    FrozenAdaptiveSelection,
    FrozenBudget,
    FrozenBudgets,
    FrozenFixedSelection,
)
from adaptive_inference.config.budgets import Budget
from adaptive_inference.experiment.frozen_inputs import (
    derive_calibration_reparse_rate,
    frozen_to_budget,
    model_cfg_for,
)


def _fb(
    *,
    low=6,
    high=8,
    measured=6.0,
    fix2=6,
    fix8=6,
    adapter="stub",
) -> FrozenBudgets:
    def mk(name, tiles, model):
        return FrozenBudget(
            name=name, max_tiles=tiles, model_name=model, adapter_kind=adapter
        )

    return FrozenBudgets(
        run_id="phase5_test",
        generated_at="2026-04-23T00:00:00+00:00",
        calibration_split_manifest="x",
        calibration_split_sha256="y",
        calibration_config_path="z",
        matched_cost_tolerance=0.1,
        b_low=mk("B_low", low, "internvl2-2b"),
        b_high=mk("B_high", high, "internvl2-2b"),
        b_fix_2b=mk("B_fix_2B", fix2, "internvl2-2b"),
        b_fix_8b=mk("B_fix_8B", fix8, "internvl2-8b"),
        adaptive_selection=FrozenAdaptiveSelection(
            target_cost_tiles=measured,
            measured_cost_tiles=measured,
            low_max_tiles=low,
            high_max_tiles=high,
        ),
        fixed_2b_selection=FrozenFixedSelection(
            target_cost_tiles=measured,
            measured_cost_tiles=measured,
            max_tiles=fix2,
            within_tolerance=True,
        ),
        fixed_8b_selection=FrozenFixedSelection(
            target_cost_tiles=measured,
            measured_cost_tiles=measured,
            max_tiles=fix8,
            within_tolerance=True,
        ),
    )


def test_frozen_to_budget_carries_name_and_tiles() -> None:
    fb = FrozenBudget(
        name="B_low", max_tiles=6, model_name="internvl2-2b", adapter_kind="stub"
    )
    budget = frozen_to_budget(fb)
    assert budget == Budget(name="B_low", max_tiles=6)


def test_reparse_rate_zero_when_measured_equals_low() -> None:
    rate = derive_calibration_reparse_rate(_fb(low=6, high=8, measured=6.0))
    assert rate.probability == 0.0
    assert rate.degenerate is True


def test_reparse_rate_positive_when_measured_above_low() -> None:
    # If reparse rate is 0.25, measured = 6 + 0.25 * 8 = 8.0
    rate = derive_calibration_reparse_rate(_fb(low=6, high=8, measured=8.0))
    assert rate.probability == pytest.approx(0.25)
    assert rate.degenerate is False


def test_reparse_rate_one_when_measured_equals_low_plus_high() -> None:
    rate = derive_calibration_reparse_rate(_fb(low=6, high=8, measured=14.0))
    assert rate.probability == pytest.approx(1.0)
    assert rate.degenerate is False


def test_reparse_rate_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="outside"):
        derive_calibration_reparse_rate(_fb(low=6, high=8, measured=5.0))
    with pytest.raises(ValueError, match="outside"):
        derive_calibration_reparse_rate(_fb(low=6, high=8, measured=20.0))


def test_reparse_rate_zero_high_raises() -> None:
    with pytest.raises(ValueError, match="non-positive"):
        derive_calibration_reparse_rate(_fb(low=6, high=0, measured=6.0))


def test_model_cfg_for_matches_adapter_kind(tmp_path: Path) -> None:
    model_yaml = tmp_path / "models.yaml"
    model_yaml.write_text(
        "models:\n"
        "  internvl2-2b:\n"
        "    adapter_kind: stub\n"
        "    model_id: OpenGVLab/InternVL2-2B\n"
        "    notes: test\n",
        encoding="utf-8",
    )
    fb = FrozenBudget(
        name="B_low", max_tiles=6, model_name="internvl2-2b", adapter_kind="stub"
    )
    cfg = model_cfg_for(fb, str(model_yaml))
    assert cfg.name == "internvl2-2b"
    assert cfg.adapter_kind == "stub"
    assert cfg.model_id == "OpenGVLab/InternVL2-2B"


def test_model_cfg_for_rejects_adapter_kind_mismatch(tmp_path: Path) -> None:
    model_yaml = tmp_path / "models.yaml"
    model_yaml.write_text(
        "models:\n"
        "  internvl2-2b:\n"
        "    adapter_kind: internvl2\n"
        "    model_id: OpenGVLab/InternVL2-2B\n",
        encoding="utf-8",
    )
    fb = FrozenBudget(
        name="B_low", max_tiles=6, model_name="internvl2-2b", adapter_kind="stub"
    )
    with pytest.raises(ValueError, match="adapter_kind"):
        model_cfg_for(fb, str(model_yaml))
