"""Tests for the model registry YAML loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_inference.config.models import load_model_config, load_model_configs


def test_load_internvl2_registry(configs_dir: Path) -> None:
    models = load_model_configs(configs_dir / "models" / "internvl2.yaml")
    assert set(models.keys()) == {"internvl2-2b", "internvl2-8b"}
    assert models["internvl2-2b"].adapter_kind == "stub"
    assert models["internvl2-2b"].model_id  # non-empty


def test_load_single_model(configs_dir: Path) -> None:
    m = load_model_config(configs_dir / "models" / "internvl2.yaml", "internvl2-8b")
    assert m.name == "internvl2-8b"
    assert m.adapter_kind == "stub"


def test_missing_model_raises(configs_dir: Path) -> None:
    with pytest.raises(KeyError, match="Model 'not-real' not found"):
        load_model_config(configs_dir / "models" / "internvl2.yaml", "not-real")


def test_missing_required_fields(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("models:\n  foo:\n    adapter_kind: stub\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required fields"):
        load_model_configs(bad)
