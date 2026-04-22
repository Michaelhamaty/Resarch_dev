"""Tests for the run-config loader (ties model + budget + prompt together)."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_inference.config.runs import load_run_config


def test_load_smoke_run_config(configs_dir: Path) -> None:
    cfg = load_run_config(configs_dir / "runs" / "smoke_single_pass.yaml")
    assert cfg.run_id == "smoke_2b_low_v1"
    assert cfg.model_cfg.name == "internvl2-2b"
    assert cfg.budget.name == "low"
    assert cfg.prompt.id == "table_parse_v1"
    assert cfg.split_name == "calibration_split"
    assert cfg.manifest_path.name == "calibration_split.json"


def test_missing_top_level_key(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("run_id: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required key"):
        load_run_config(bad)
