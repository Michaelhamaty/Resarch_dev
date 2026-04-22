"""Tests for the versioned prompt template loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_inference.config.prompts import PromptTemplate, load_prompt_template


def test_load_table_parse_v1(configs_dir: Path) -> None:
    prompt = load_prompt_template(configs_dir / "prompts" / "table_parse_v1.yaml")
    assert prompt.id == "table_parse_v1"
    assert prompt.version == 1
    assert "HTML" in prompt.template
    assert prompt.description  # non-empty


def test_missing_required_field_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("id: x\nversion: 1\ndescription: y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required fields"):
        load_prompt_template(bad)


def test_non_mapping_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        load_prompt_template(bad)


def test_frozen(configs_dir: Path) -> None:
    prompt = load_prompt_template(configs_dir / "prompts" / "table_parse_v1.yaml")
    with pytest.raises(Exception):
        prompt.version = 2  # type: ignore[misc]


def test_from_mapping_coerces_types() -> None:
    p = PromptTemplate.from_mapping(
        {"id": "t", "version": "3", "description": "d", "template": "hi"}
    )
    assert p.version == 3
