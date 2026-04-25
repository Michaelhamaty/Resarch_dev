"""Unit tests for the Phase 6 YAML config loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_inference.experiment.config import load_phase6_config


_GOOD_YAML = """
run_set_id: phase6_main_v1
frozen_budgets_path: configs/calibration/frozen_budgets.json
inputs:
  split_name: held_out_eval_split
  manifest_path: data/splits/held_out_eval_split.json
  records_path: data/fixtures/sample_pages.json
  image_root: data
model:
  config_path: configs/models/internvl2.yaml
prompt:
  config_path: configs/prompts/table_parse_v1.yaml
output:
  root: outputs/runs/phase6
random_escalation:
  seeds: [0, 1, 2]
"""


def test_load_phase6_config_happy_path(tmp_path: Path) -> None:
    p = tmp_path / "phase6.yaml"
    p.write_text(_GOOD_YAML, encoding="utf-8")
    cfg = load_phase6_config(p)
    assert cfg.run_set_id == "phase6_main_v1"
    assert cfg.split_name == "held_out_eval_split"
    assert cfg.random_seeds == (0, 1, 2)
    assert cfg.frozen_budgets_path == Path("configs/calibration/frozen_budgets.json")
    assert cfg.output_root == Path("outputs/runs/phase6")


def test_load_phase6_config_missing_key_raises(tmp_path: Path) -> None:
    p = tmp_path / "phase6.yaml"
    p.write_text(
        "run_set_id: x\ninputs: {}\nmodel: {}\nprompt: {}\noutput: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="frozen_budgets_path"):
        load_phase6_config(p)


def test_load_phase6_config_default_seeds(tmp_path: Path) -> None:
    yaml_without_seeds = _GOOD_YAML.replace("random_escalation:\n  seeds: [0, 1, 2]\n", "")
    p = tmp_path / "phase6.yaml"
    p.write_text(yaml_without_seeds, encoding="utf-8")
    cfg = load_phase6_config(p)
    assert cfg.random_seeds == (0, 1, 2)


def test_load_phase6_config_rejects_non_int_seeds(tmp_path: Path) -> None:
    bad = _GOOD_YAML.replace("[0, 1, 2]", "[0, 'x']")
    p = tmp_path / "phase6.yaml"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="list of ints"):
        load_phase6_config(p)
