"""Frozen calibration artifact: the only Phase-5 output Phase-6 consumes.

Schema (``schema_version=1``):

```json
{
  "schema_version": 1,
  "run_id": "phase5_calibration_v1",
  "generated_at": "2026-04-23T12:34:56.789012+00:00",
  "pinning": {
    "calibration_split_manifest": "data/splits/calibration_split.json",
    "calibration_split_sha256": "…",
    "calibration_config_path": "configs/calibration/phase5.yaml",
    "cost_unit": "max_tiles",
    "matched_cost_tolerance": 0.10
  },
  "budgets": {
    "B_low":     {"name": "B_low",     "max_tiles": 4, "model_name": "internvl2-2b", "adapter_kind": "stub"},
    "B_high":    {"name": "B_high",    "max_tiles": 12, ...},
    "B_fix_2B":  {"name": "B_fix_2B",  "max_tiles": 6,  ...},
    "B_fix_8B":  {"name": "B_fix_8B",  "max_tiles": 6,  ...}
  },
  "selection": {
    "adaptive": {
      "target_cost_tiles": 6.0,
      "measured_cost_tiles": 4.0,
      "low_max_tiles": 4, "high_max_tiles": 12
    },
    "fixed_2b": {"target_cost_tiles": 4.0, "measured_cost_tiles": 4.0, "max_tiles": 4, "within_tolerance": true},
    "fixed_8b": {"target_cost_tiles": 4.0, "measured_cost_tiles": 4.0, "max_tiles": 4, "within_tolerance": true}
  }
}
```

The four ``budgets`` entries are what Phase 6 run configs reference
symbolically. Sweep-point statistics live alongside in
``outputs/calibration/sweep_summaries.jsonl``; they are *not* in this
artifact because they would make it large and change frequently.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

SCHEMA_VERSION = 1
COST_UNIT = "max_tiles"


@dataclass(frozen=True)
class FrozenBudget:
    name: str
    max_tiles: int
    model_name: str
    adapter_kind: str


@dataclass(frozen=True)
class FrozenAdaptiveSelection:
    target_cost_tiles: float
    measured_cost_tiles: float
    low_max_tiles: int
    high_max_tiles: int


@dataclass(frozen=True)
class FrozenFixedSelection:
    target_cost_tiles: float
    measured_cost_tiles: float
    max_tiles: int
    within_tolerance: bool


@dataclass(frozen=True)
class FrozenBudgets:
    run_id: str
    generated_at: str
    calibration_split_manifest: str
    calibration_split_sha256: str
    calibration_config_path: str
    matched_cost_tolerance: float
    b_low: FrozenBudget
    b_high: FrozenBudget
    b_fix_2b: FrozenBudget
    b_fix_8b: FrozenBudget
    adaptive_selection: FrozenAdaptiveSelection
    fixed_2b_selection: FrozenFixedSelection
    fixed_8b_selection: FrozenFixedSelection


def write_frozen_budgets(path: str | Path, fb: FrozenBudgets) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": fb.run_id,
        "generated_at": fb.generated_at,
        "pinning": {
            "calibration_split_manifest": fb.calibration_split_manifest,
            "calibration_split_sha256": fb.calibration_split_sha256,
            "calibration_config_path": fb.calibration_config_path,
            "cost_unit": COST_UNIT,
            "matched_cost_tolerance": fb.matched_cost_tolerance,
        },
        "budgets": {
            "B_low": asdict(fb.b_low),
            "B_high": asdict(fb.b_high),
            "B_fix_2B": asdict(fb.b_fix_2b),
            "B_fix_8B": asdict(fb.b_fix_8b),
        },
        "selection": {
            "adaptive": asdict(fb.adaptive_selection),
            "fixed_2b": asdict(fb.fixed_2b_selection),
            "fixed_8b": asdict(fb.fixed_8b_selection),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def load_frozen_budgets(path: str | Path) -> FrozenBudgets:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Frozen budgets at {path} has schema_version={raw.get('schema_version')},"
            f" expected {SCHEMA_VERSION}"
        )

    pinning = raw["pinning"]
    if pinning.get("cost_unit") != COST_UNIT:
        raise ValueError(
            f"Frozen budgets at {path} has cost_unit={pinning.get('cost_unit')},"
            f" expected {COST_UNIT}"
        )

    budgets = raw["budgets"]
    selection = raw["selection"]

    return FrozenBudgets(
        run_id=str(raw["run_id"]),
        generated_at=str(raw["generated_at"]),
        calibration_split_manifest=str(pinning["calibration_split_manifest"]),
        calibration_split_sha256=str(pinning["calibration_split_sha256"]),
        calibration_config_path=str(pinning["calibration_config_path"]),
        matched_cost_tolerance=float(pinning["matched_cost_tolerance"]),
        b_low=FrozenBudget(**budgets["B_low"]),
        b_high=FrozenBudget(**budgets["B_high"]),
        b_fix_2b=FrozenBudget(**budgets["B_fix_2B"]),
        b_fix_8b=FrozenBudget(**budgets["B_fix_8B"]),
        adaptive_selection=FrozenAdaptiveSelection(**selection["adaptive"]),
        fixed_2b_selection=FrozenFixedSelection(**selection["fixed_2b"]),
        fixed_8b_selection=FrozenFixedSelection(**selection["fixed_8b"]),
    )
