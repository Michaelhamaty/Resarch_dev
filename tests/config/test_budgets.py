"""Tests for the named-budget YAML loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_inference.config.budgets import load_budget, load_budgets


def test_load_phase2_budgets(configs_dir: Path) -> None:
    budgets = load_budgets(configs_dir / "budgets" / "phase2.yaml")
    assert set(budgets.keys()) == {"low", "high"}
    assert budgets["low"].max_tiles < budgets["high"].max_tiles
    assert budgets["low"].name == "low"


def test_load_single_budget(configs_dir: Path) -> None:
    b = load_budget(configs_dir / "budgets" / "phase2.yaml", "low")
    assert b.name == "low"
    assert b.max_tiles > 0


def test_missing_budget_raises(configs_dir: Path) -> None:
    with pytest.raises(KeyError, match="Budget 'medium' not found"):
        load_budget(configs_dir / "budgets" / "phase2.yaml", "medium")


def test_rejects_non_positive(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("budgets:\n  bad:\n    max_tiles: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-positive max_tiles"):
        load_budgets(bad)


def test_rejects_missing_section(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("something: else\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'budgets' key"):
        load_budgets(bad)
