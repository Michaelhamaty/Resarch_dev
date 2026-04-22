"""Adaptive run configuration: one model + prompt + *two* budgets.

The adaptive pipeline (Phase 4) needs two named budgets — ``budget_low``
for the first pass and ``budget_high`` for the optional one-shot reparse
— plus a single model and the fixed prompt. Everything else mirrors the
Phase 2 ``RunConfig`` so the smoke path keeps the same directory shape
for manifests and image roots.

YAML shape:

```yaml
run_id: smoke_2b_adaptive_v1
inputs:
  split_name: calibration_split
  manifest_path: data/splits/calibration_split.json
  records_path: data/fixtures/sample_pages.json
  image_root: data
model:
  config_path: configs/models/internvl2.yaml
  name: internvl2-2b
budget:
  config_path: configs/budgets/phase2.yaml
  low_name: low
  high_name: high
prompt:
  config_path: configs/prompts/table_parse_v1.yaml
output:
  dir: outputs/runs/smoke_2b_adaptive_v1
```
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
class AdaptiveRunConfig:
    run_id: str
    split_name: str
    manifest_path: Path
    records_path: Path
    image_root: Path
    model_cfg: ModelConfig
    budget_low: Budget
    budget_high: Budget
    prompt: PromptTemplate
    output_dir: Path


def load_adaptive_run_config(path: str | Path) -> AdaptiveRunConfig:
    """Load an adaptive run YAML and resolve every referenced sub-config."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Adaptive run config at {path} must be a YAML mapping")

    for key in _REQUIRED_TOP_LEVEL:
        if key not in raw:
            raise ValueError(
                f"Adaptive run config at {path} missing required key: {key}"
            )

    model_section = _require_mapping(raw, "model", path)
    budget_section = _require_mapping(raw, "budget", path)
    prompt_section = _require_mapping(raw, "prompt", path)
    inputs_section = _require_mapping(raw, "inputs", path)
    output_section = _require_mapping(raw, "output", path)

    for key in ("low_name", "high_name", "config_path"):
        if key not in budget_section:
            raise ValueError(
                f"Adaptive run config at {path} budget section missing: {key}"
            )

    model_cfg = load_model_config(model_section["config_path"], model_section["name"])
    budget_low = load_budget(budget_section["config_path"], budget_section["low_name"])
    budget_high = load_budget(budget_section["config_path"], budget_section["high_name"])
    prompt = load_prompt_template(prompt_section["config_path"])

    return AdaptiveRunConfig(
        run_id=str(raw["run_id"]),
        split_name=str(inputs_section.get("split_name", "unknown")),
        manifest_path=Path(inputs_section["manifest_path"]),
        records_path=Path(inputs_section["records_path"]),
        image_root=Path(inputs_section["image_root"]),
        model_cfg=model_cfg,
        budget_low=budget_low,
        budget_high=budget_high,
        prompt=prompt,
        output_dir=Path(output_section["dir"]),
    )


_REQUIRED_TOP_LEVEL = ("run_id", "model", "budget", "prompt", "inputs", "output")


def _require_mapping(raw: dict[str, Any], key: str, path: str | Path) -> dict[str, Any]:
    section = raw[key]
    if not isinstance(section, dict):
        raise ValueError(
            f"Adaptive run config at {path} has non-mapping section: {key}"
        )
    return section
