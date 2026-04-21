"""Minimal YAML -> typed-config loader. One function per schema."""

from pathlib import Path
from typing import Any

import yaml

from adaptive_inference.config.schemas import (
    BudgetConfig,
    ModelConfig,
    PromptConfig,
    RunConfig,
)


def load_yaml_file(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a mapping at the top of {path}, got {type(data).__name__}."
        )
    return data


def load_model_config(path: Path) -> ModelConfig:
    return ModelConfig(**load_yaml_file(path))


def load_prompt_config(path: Path) -> PromptConfig:
    return PromptConfig(**load_yaml_file(path))


def load_budgets_file(path: Path) -> dict[str, BudgetConfig]:
    """Load a YAML file shaped as ``{budgets: [ {...}, {...} ]}`` into a name->config dict."""
    data = load_yaml_file(path)
    entries = data.get("budgets")
    if not isinstance(entries, list):
        raise ValueError(f"{path} must contain a top-level 'budgets' list.")
    budgets: dict[str, BudgetConfig] = {}
    for entry in entries:
        cfg = BudgetConfig(**entry)
        budgets[cfg.name] = cfg
    return budgets


def load_run_config(path: Path) -> RunConfig:
    return RunConfig(**load_yaml_file(path))
