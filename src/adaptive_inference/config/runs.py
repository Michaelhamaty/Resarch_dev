"""Run configuration: ties a manifest to a model + budget + prompt + output dir.

A ``RunConfig`` is the single object the smoke runner consumes. The YAML
form references the other configs by path, so re-running an experiment is
a matter of pointing at the same run config — model/prompt/budget are
resolved transitively and pinned in logs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .budgets import Budget, load_budget
from .models import ModelConfig, load_model_config
from .prompts import PromptTemplate, load_prompt_template


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    split_name: str
    manifest_path: Path
    records_path: Path
    image_root: Path
    model_cfg: ModelConfig
    budget: Budget
    prompt: PromptTemplate
    output_dir: Path


def load_run_config(path: str | Path) -> RunConfig:
    """Load a run YAML and resolve every referenced sub-config."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Run config at {path} must be a YAML mapping")

    for key in _REQUIRED_TOP_LEVEL:
        if key not in raw:
            raise ValueError(f"Run config at {path} missing required key: {key}")

    model_section = _require_mapping(raw, "model", path)
    budget_section = _require_mapping(raw, "budget", path)
    prompt_section = _require_mapping(raw, "prompt", path)
    inputs_section = _require_mapping(raw, "inputs", path)
    output_section = _require_mapping(raw, "output", path)

    model_cfg = load_model_config(model_section["config_path"], model_section["name"])
    budget = load_budget(budget_section["config_path"], budget_section["name"])
    prompt = load_prompt_template(prompt_section["config_path"])

    return RunConfig(
        run_id=str(raw["run_id"]),
        split_name=str(inputs_section.get("split_name", "unknown")),
        manifest_path=Path(inputs_section["manifest_path"]),
        records_path=Path(inputs_section["records_path"]),
        image_root=Path(inputs_section["image_root"]),
        model_cfg=model_cfg,
        budget=budget,
        prompt=prompt,
        output_dir=Path(output_section["dir"]),
    )


_REQUIRED_TOP_LEVEL = ("run_id", "model", "budget", "prompt", "inputs", "output")


def _require_mapping(raw: dict[str, Any], key: str, path: str | Path) -> dict[str, Any]:
    section = raw[key]
    if not isinstance(section, dict):
        raise ValueError(f"Run config at {path} has non-mapping section: {key}")
    return section
