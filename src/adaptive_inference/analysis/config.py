"""Phase 7 configuration loader.

The Phase 7 YAML points at the four input files Phase 7 needs to read
and the output root it should write under. Everything else (random
seeds, system lists, prompt id) is derived from the Phase 6 manifest
itself, so there is exactly one source of truth per piece of metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml


@dataclass(frozen=True)
class Phase7Config:
    """Loaded Phase 7 YAML."""

    phase6_manifest_path: Path
    frozen_budgets_path: Path
    held_out_split_path: Path
    calibration_split_path: Path
    output_root: Path
    scoring_summaries_root: Path | None = None
    splits_are_identical_acknowledged: bool = False


def load_phase7_config(path: str | Path) -> Phase7Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: Phase 7 config must be a YAML mapping")

    inputs = _require(raw, "inputs", path)
    output = _require(raw, "output", path)

    scoring_root_raw = inputs.get("scoring_summaries_root") if isinstance(inputs, dict) else None
    scoring_root = Path(scoring_root_raw) if scoring_root_raw else None
    splits_identical = bool(inputs.get("splits_are_identical_acknowledged", False)) if isinstance(inputs, dict) else False

    return Phase7Config(
        phase6_manifest_path=Path(_require(inputs, "phase6_manifest_path", path)),
        frozen_budgets_path=Path(_require(inputs, "frozen_budgets_path", path)),
        held_out_split_path=Path(_require(inputs, "held_out_split_path", path)),
        calibration_split_path=Path(_require(inputs, "calibration_split_path", path)),
        output_root=Path(_require(output, "root", path)),
        scoring_summaries_root=scoring_root,
        splits_are_identical_acknowledged=splits_identical,
    )


def _require(d: Mapping[str, object], key: str, path: object) -> object:
    if key not in d:
        raise ValueError(f"{path}: Phase 7 config missing required key {key!r}")
    return d[key]
