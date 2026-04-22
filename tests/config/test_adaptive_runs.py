"""Loader tests for ``AdaptiveRunConfig`` (Phase 4 run config)."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_inference.config.adaptive_runs import load_adaptive_run_config


def test_loads_smoke_adaptive_config(configs_dir: Path) -> None:
    cfg = load_adaptive_run_config(configs_dir / "runs" / "smoke_adaptive.yaml")
    assert cfg.run_id == "smoke_2b_adaptive_v1"
    assert cfg.split_name == "calibration_split"
    assert cfg.model_cfg.name == "internvl2-2b"
    assert cfg.budget_low.name == "low"
    assert cfg.budget_high.name == "high"
    assert cfg.prompt.id == "table_parse_v1"


def test_missing_budget_name_errors(tmp_path: Path, configs_dir: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """run_id: r
inputs:
  split_name: s
  manifest_path: m
  records_path: r
  image_root: i
model:
  config_path: {models}
  name: internvl2-2b
budget:
  config_path: {budgets}
  low_name: low
prompt:
  config_path: {prompt}
output:
  dir: out
""".format(
            models=configs_dir / "models" / "internvl2.yaml",
            budgets=configs_dir / "budgets" / "phase2.yaml",
            prompt=configs_dir / "prompts" / "table_parse_v1.yaml",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="high_name"):
        load_adaptive_run_config(bad)


def test_non_mapping_errors(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- not_a_mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_adaptive_run_config(bad)
