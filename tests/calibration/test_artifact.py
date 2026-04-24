"""Unit tests for FrozenBudgets read/write round-trip + schema guard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_inference.calibration.artifact import (
    FrozenAdaptiveSelection,
    FrozenBudget,
    FrozenBudgets,
    FrozenFixedSelection,
    load_frozen_budgets,
    write_frozen_budgets,
)


def _sample(run_id: str = "test_run") -> FrozenBudgets:
    return FrozenBudgets(
        run_id=run_id,
        generated_at="2026-04-23T00:00:00+00:00",
        calibration_split_manifest="data/splits/calibration_split.json",
        calibration_split_sha256="deadbeef" * 8,
        calibration_config_path="configs/calibration/phase5.yaml",
        matched_cost_tolerance=0.10,
        b_low=FrozenBudget(
            name="B_low", max_tiles=4, model_name="internvl2-2b", adapter_kind="stub"
        ),
        b_high=FrozenBudget(
            name="B_high", max_tiles=12, model_name="internvl2-2b", adapter_kind="stub"
        ),
        b_fix_2b=FrozenBudget(
            name="B_fix_2B", max_tiles=6, model_name="internvl2-2b", adapter_kind="stub"
        ),
        b_fix_8b=FrozenBudget(
            name="B_fix_8B", max_tiles=6, model_name="internvl2-8b", adapter_kind="stub"
        ),
        adaptive_selection=FrozenAdaptiveSelection(
            target_cost_tiles=6.0,
            measured_cost_tiles=4.0,
            low_max_tiles=4,
            high_max_tiles=12,
        ),
        fixed_2b_selection=FrozenFixedSelection(
            target_cost_tiles=4.0,
            measured_cost_tiles=4.0,
            max_tiles=4,
            within_tolerance=True,
        ),
        fixed_8b_selection=FrozenFixedSelection(
            target_cost_tiles=4.0,
            measured_cost_tiles=4.0,
            max_tiles=4,
            within_tolerance=True,
        ),
    )


def test_round_trip_preserves_all_fields(tmp_path: Path) -> None:
    fb = _sample()
    path = tmp_path / "frozen.json"
    write_frozen_budgets(path, fb)
    loaded = load_frozen_budgets(path)
    assert loaded == fb


def test_write_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "frozen.json"
    write_frozen_budgets(path, _sample())
    assert path.exists()


def test_on_disk_layout_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "frozen.json"
    write_frozen_budgets(path, _sample())
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert set(payload["budgets"].keys()) == {"B_low", "B_high", "B_fix_2B", "B_fix_8B"}
    assert payload["pinning"]["cost_unit"] == "max_tiles"


def test_load_rejects_bad_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    write_frozen_budgets(path, _sample())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 999
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_frozen_budgets(path)


def test_load_rejects_bad_cost_unit(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    write_frozen_budgets(path, _sample())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["pinning"]["cost_unit"] = "runtime_ms"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_frozen_budgets(path)
