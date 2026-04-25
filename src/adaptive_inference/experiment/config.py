"""YAML loader for the Phase 6 top-level experiment config.

Keeps orchestration configuration separate from individual run configs
(Phase 2/4 ones) because Phase 6 points at four different budgets via
the frozen artifact, not a per-run ``budget: {config_path, name}``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .runner import Phase6Config


def load_phase6_config(path: str | Path) -> Phase6Config:
    """Parse a Phase 6 YAML into a ``Phase6Config``.

    The shape is small enough to be validated inline rather than through
    a generic schema library.
    """

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Phase 6 config at {path} must be a YAML mapping")

    for key in ("run_set_id", "frozen_budgets_path", "inputs", "model", "prompt", "output"):
        if key not in raw:
            raise ValueError(f"Phase 6 config at {path} missing required key: {key}")

    inputs = _require_mapping(raw, "inputs", path)
    model = _require_mapping(raw, "model", path)
    prompt = _require_mapping(raw, "prompt", path)
    output = _require_mapping(raw, "output", path)

    random_section = raw.get("random_escalation") or {}
    if not isinstance(random_section, dict):
        raise ValueError(
            f"Phase 6 config at {path} 'random_escalation' must be a mapping"
        )
    seeds_raw = random_section.get("seeds", [0, 1, 2])
    if not isinstance(seeds_raw, list) or not all(isinstance(s, int) for s in seeds_raw):
        raise ValueError(
            f"Phase 6 config at {path} 'random_escalation.seeds' must be a list of ints"
        )
    seeds = tuple(seeds_raw)

    return Phase6Config(
        run_set_id=str(raw["run_set_id"]),
        frozen_budgets_path=Path(raw["frozen_budgets_path"]),
        split_name=str(inputs.get("split_name", "held_out_eval_split")),
        held_out_manifest_path=Path(inputs["manifest_path"]),
        records_path=Path(inputs["records_path"]),
        image_root=Path(inputs["image_root"]),
        model_config_path=Path(model["config_path"]),
        prompt_config_path=Path(prompt["config_path"]),
        output_root=Path(output["root"]),
        random_seeds=seeds,
    )


def _require_mapping(raw: dict[str, Any], key: str, path: str | Path) -> dict[str, Any]:
    section = raw[key]
    if not isinstance(section, dict):
        raise ValueError(
            f"Phase 6 config at {path} has non-mapping section: {key}"
        )
    return section
