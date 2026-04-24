"""Calibration config: candidate grids, target cost, tolerance, paths.

YAML shape (see ``configs/calibration/phase5.yaml``):

```yaml
run_id: phase5_calibration_v1
split:
  manifest_path: data/splits/calibration_split.json
  records_path: data/fixtures/sample_pages.json
  image_root: data
prompt_config_path: configs/prompts/table_parse_v1.yaml
model_config_path: configs/models/internvl2.yaml
candidates:
  low_max_tiles: [2, 4, 6]
  high_max_tiles: [8, 12, 16]
  fixed_2b_max_tiles: [4, 6, 8, 10, 12]
  fixed_8b_max_tiles: [4, 6, 8, 10, 12]
target_adaptive_cost_tiles: 6.0
matched_cost_tolerance: 0.10
models:
  adaptive: internvl2-2b
  fixed_2b: internvl2-2b
  fixed_8b: internvl2-8b
output:
  sweep_root: outputs/calibration/sweep
  sweep_summaries_path: outputs/calibration/sweep_summaries.jsonl
  frozen_budgets_path: configs/calibration/frozen_budgets.json
```

The loader validates structure but does not yet open the referenced
sub-configs — that happens inside the sweep code so missing files
surface with the stage-appropriate error message.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CalibrationConfig:
    run_id: str
    manifest_path: Path
    records_path: Path
    image_root: Path
    prompt_config_path: Path
    model_config_path: Path
    low_max_tiles: tuple[int, ...]
    high_max_tiles: tuple[int, ...]
    fixed_2b_max_tiles: tuple[int, ...]
    fixed_8b_max_tiles: tuple[int, ...]
    target_adaptive_cost_tiles: float
    matched_cost_tolerance: float
    adaptive_model_name: str
    fixed_2b_model_name: str
    fixed_8b_model_name: str
    sweep_root: Path
    sweep_summaries_path: Path
    frozen_budgets_path: Path


_REQUIRED_TOP_LEVEL = (
    "run_id",
    "split",
    "prompt_config_path",
    "model_config_path",
    "candidates",
    "target_adaptive_cost_tiles",
    "matched_cost_tolerance",
    "models",
    "output",
)


def load_calibration_config(path: str | Path) -> CalibrationConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Calibration config at {path} must be a YAML mapping")

    for key in _REQUIRED_TOP_LEVEL:
        if key not in raw:
            raise ValueError(
                f"Calibration config at {path} missing required key: {key}"
            )

    split = _require_mapping(raw, "split", path)
    candidates = _require_mapping(raw, "candidates", path)
    models = _require_mapping(raw, "models", path)
    output = _require_mapping(raw, "output", path)

    for field in ("manifest_path", "records_path", "image_root"):
        if field not in split:
            raise ValueError(
                f"Calibration config at {path} split section missing: {field}"
            )

    tolerance = float(raw["matched_cost_tolerance"])
    if tolerance < 0:
        raise ValueError(
            f"matched_cost_tolerance must be non-negative, got {tolerance}"
        )

    target = float(raw["target_adaptive_cost_tiles"])
    if target <= 0:
        raise ValueError(
            f"target_adaptive_cost_tiles must be positive, got {target}"
        )

    return CalibrationConfig(
        run_id=str(raw["run_id"]),
        manifest_path=Path(split["manifest_path"]),
        records_path=Path(split["records_path"]),
        image_root=Path(split["image_root"]),
        prompt_config_path=Path(raw["prompt_config_path"]),
        model_config_path=Path(raw["model_config_path"]),
        low_max_tiles=_require_positive_int_list(candidates, "low_max_tiles", path),
        high_max_tiles=_require_positive_int_list(candidates, "high_max_tiles", path),
        fixed_2b_max_tiles=_require_positive_int_list(
            candidates, "fixed_2b_max_tiles", path
        ),
        fixed_8b_max_tiles=_require_positive_int_list(
            candidates, "fixed_8b_max_tiles", path
        ),
        target_adaptive_cost_tiles=target,
        matched_cost_tolerance=tolerance,
        adaptive_model_name=_require_str(models, "adaptive", path),
        fixed_2b_model_name=_require_str(models, "fixed_2b", path),
        fixed_8b_model_name=_require_str(models, "fixed_8b", path),
        sweep_root=Path(_require_str(output, "sweep_root", path)),
        sweep_summaries_path=Path(_require_str(output, "sweep_summaries_path", path)),
        frozen_budgets_path=Path(_require_str(output, "frozen_budgets_path", path)),
    )


def _require_mapping(raw: dict[str, Any], key: str, path: str | Path) -> dict[str, Any]:
    section = raw[key]
    if not isinstance(section, dict):
        raise ValueError(
            f"Calibration config at {path} has non-mapping section: {key}"
        )
    return section


def _require_str(section: dict[str, Any], key: str, path: str | Path) -> str:
    if key not in section:
        raise ValueError(f"Calibration config at {path} missing field: {key}")
    return str(section[key])


def _require_positive_int_list(
    section: dict[str, Any], key: str, path: str | Path
) -> tuple[int, ...]:
    if key not in section:
        raise ValueError(
            f"Calibration config at {path} candidates missing: {key}"
        )
    raw = section[key]
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            f"Calibration config at {path}: {key} must be a non-empty list"
        )
    out: list[int] = []
    for v in raw:
        iv = int(v)
        if iv <= 0:
            raise ValueError(
                f"Calibration config at {path}: {key} entries must be positive, got {v}"
            )
        out.append(iv)
    return tuple(out)
