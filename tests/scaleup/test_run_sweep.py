"""Pure-logic tests for scripts/scaleup/run_sweep.py.

The GPU-touching part (calling ``run_phase6``) is exercised by Stage 7
on the VM. These tests cover the config parsing, per-dataset
synthesis, and resume-skip logic.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml


def _load_sweep_module():
    repo_root = Path(__file__).resolve().parents[2]
    spec_path = repo_root / "scripts" / "scaleup" / "run_sweep.py"
    spec = importlib.util.spec_from_file_location("run_sweep", spec_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_sweep"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sweep():
    return _load_sweep_module()


def _write_sweep_config(tmp_path: Path, *, datasets: dict, seeds=(0, 1, 2)) -> Path:
    payload = {
        "run_set_id": "test_sweep",
        "datasets": datasets,
        "model": {"config_path": "configs/models/internvl2_real.yaml"},
        "prompt": {"config_path": "configs/prompts/table_parse_v2.yaml"},
        "output": {"root": "outputs/scaleup_v2"},
        "random_escalation": {"seeds": list(seeds)},
    }
    p = tmp_path / "scaleup_v2.yaml"
    # default_flow_style=False keeps the YAML block format; sort_keys=False
    # preserves the caller's dataset order so order-sensitive tests work.
    p.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return p


def test_load_sweep_config_two_datasets(sweep, tmp_path):
    cfg_path = _write_sweep_config(tmp_path, datasets={
        "omnidocbench": {
            "split_manifest_path": "data/splits/scaleup_v2/omnidocbench/held_out.json",
            "records_path": "data/omnidocbench/records.json",
            "image_root": "data",
            "frozen_budgets_path": "configs/calibration/frozen_budgets_v2_omnidocbench.json",
        },
        "fintabnet": {
            "split_manifest_path": "data/splits/scaleup_v2/fintabnet/held_out_100.json",
            "records_path": "data/fintabnet/records.json",
            "image_root": "data",
            "frozen_budgets_path": "configs/calibration/frozen_budgets_v2_fintabnet.json",
        },
    })
    cfg = sweep.load_sweep_config(cfg_path)
    assert cfg.run_set_id == "test_sweep"
    assert [d.name for d in cfg.datasets] == ["omnidocbench", "fintabnet"]
    assert cfg.datasets[1].split_manifest_path == Path(
        "data/splits/scaleup_v2/fintabnet/held_out_100.json"
    )
    assert cfg.random_seeds == (0, 1, 2)


def test_load_sweep_config_rejects_missing_required_keys(sweep, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"run_set_id": "x"}), encoding="utf-8")
    with pytest.raises(ValueError):
        sweep.load_sweep_config(bad)


def test_load_sweep_config_rejects_missing_dataset_keys(sweep, tmp_path):
    cfg_path = _write_sweep_config(tmp_path, datasets={
        "broken": {"records_path": "x"},  # missing split_manifest_path etc.
    })
    with pytest.raises(ValueError):
        sweep.load_sweep_config(cfg_path)


def test_load_sweep_config_rejects_non_int_seeds(sweep, tmp_path):
    payload = {
        "run_set_id": "x",
        "datasets": {"a": {
            "split_manifest_path": "x", "records_path": "y",
            "image_root": "z", "frozen_budgets_path": "w",
        }},
        "model": {"config_path": "m"},
        "prompt": {"config_path": "p"},
        "output": {"root": "o"},
        "random_escalation": {"seeds": ["zero", "one"]},
    }
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        sweep.load_sweep_config(p)


def test_phase6_config_for_synthesizes_per_dataset_paths(sweep, tmp_path):
    cfg_path = _write_sweep_config(tmp_path, datasets={
        "omnidocbench": {
            "split_manifest_path": "data/splits/scaleup_v2/omnidocbench/held_out.json",
            "records_path": "data/omnidocbench/records.json",
            "image_root": "data",
            "frozen_budgets_path": "configs/calibration/frozen_budgets_v2_omnidocbench.json",
        },
    })
    cfg = sweep.load_sweep_config(cfg_path)
    p6 = sweep.phase6_config_for(cfg, cfg.datasets[0])
    assert p6.run_set_id == "test_sweep__omnidocbench"
    assert p6.output_root == Path("outputs/scaleup_v2/omnidocbench")
    assert p6.held_out_manifest_path == Path(
        "data/splits/scaleup_v2/omnidocbench/held_out.json"
    )
    assert p6.frozen_budgets_path == Path(
        "configs/calibration/frozen_budgets_v2_omnidocbench.json"
    )
    assert p6.random_seeds == (0, 1, 2)


def test_phase6_config_split_name_includes_dataset(sweep, tmp_path):
    cfg_path = _write_sweep_config(tmp_path, datasets={
        "fintabnet": {
            "split_manifest_path": "x", "records_path": "y",
            "image_root": "z", "frozen_budgets_path": "w",
        },
    })
    cfg = sweep.load_sweep_config(cfg_path)
    p6 = sweep.phase6_config_for(cfg, cfg.datasets[0])
    assert "fintabnet" in p6.split_name


def test_dataset_already_completed_true_when_manifest_exists(sweep, tmp_path):
    cfg_path = _write_sweep_config(tmp_path, datasets={
        "a": {
            "split_manifest_path": "x", "records_path": "y",
            "image_root": "z", "frozen_budgets_path": "w",
        },
    })
    cfg = sweep.load_sweep_config(cfg_path)
    p6 = sweep.phase6_config_for(cfg, cfg.datasets[0])
    p6.output_root.mkdir(parents=True, exist_ok=True)
    (p6.output_root / "manifest.json").write_text("{}", encoding="utf-8")
    assert sweep.dataset_already_completed(p6) is True


def test_dataset_already_completed_false_when_dir_missing(sweep, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_path = _write_sweep_config(tmp_path, datasets={
        "never_ran": {
            "split_manifest_path": "x", "records_path": "y",
            "image_root": "z", "frozen_budgets_path": "w",
        },
    })
    cfg = sweep.load_sweep_config(cfg_path)
    p6 = sweep.phase6_config_for(cfg, cfg.datasets[0])
    assert sweep.dataset_already_completed(p6) is False
