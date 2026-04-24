"""Phase 5 calibration CLI: run sweeps, freeze budgets, write artifacts.

Loads a calibration YAML, executes the adaptive sweep plus two
fixed-cost sweeps on the calibration split, picks ``(B_low, B_high)``
by matching the configured target cost and ``B_fix_2B`` / ``B_fix_8B``
by matching the adaptive pair's measured cost, and writes:

- ``configs/calibration/frozen_budgets.json`` — frozen artifact
  consumed by Phase 6.
- ``outputs/calibration/sweep_summaries.jsonl`` — one line per sweep
  point for audit.

Usage:

    uv run python scripts/calibration/run_calibration.py \
        --config configs/calibration/phase5.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from adaptive_inference.calibration.artifact import (
    FrozenAdaptiveSelection,
    FrozenBudget,
    FrozenBudgets,
    FrozenFixedSelection,
    write_frozen_budgets,
)
from adaptive_inference.calibration.config import (
    CalibrationConfig,
    load_calibration_config,
)
from adaptive_inference.calibration.select import (
    SelectedAdaptive,
    SelectedFixed,
    select_adaptive_pair,
    select_matched_fixed,
)
from adaptive_inference.calibration.sweep import (
    AdaptiveSweepPoint,
    FixedSweepPoint,
    sweep_adaptive,
    sweep_fixed,
)
from adaptive_inference.config.models import load_model_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a calibration YAML (see configs/calibration/phase5.yaml).",
    )
    args = parser.parse_args()

    cfg = load_calibration_config(args.config)

    adaptive_points = sweep_adaptive(cfg)
    fixed_2b_points = sweep_fixed(
        cfg,
        model_name=cfg.fixed_2b_model_name,
        candidates=cfg.fixed_2b_max_tiles,
    )
    fixed_8b_points = sweep_fixed(
        cfg,
        model_name=cfg.fixed_8b_model_name,
        candidates=cfg.fixed_8b_max_tiles,
    )

    _write_sweep_summaries(
        cfg.sweep_summaries_path,
        adaptive_points=adaptive_points,
        fixed_2b_points=fixed_2b_points,
        fixed_8b_points=fixed_8b_points,
    )

    selected_adaptive = select_adaptive_pair(
        adaptive_points, target_cost_tiles=cfg.target_adaptive_cost_tiles
    )
    target_for_fixed = selected_adaptive.measured_cost_tiles
    selected_fixed_2b = select_matched_fixed(
        fixed_2b_points,
        target_cost_tiles=target_for_fixed,
        tolerance=cfg.matched_cost_tolerance,
    )
    selected_fixed_8b = select_matched_fixed(
        fixed_8b_points,
        target_cost_tiles=target_for_fixed,
        tolerance=cfg.matched_cost_tolerance,
    )

    adaptive_model = load_model_config(cfg.model_config_path, cfg.adaptive_model_name)
    fixed_2b_model = load_model_config(cfg.model_config_path, cfg.fixed_2b_model_name)
    fixed_8b_model = load_model_config(cfg.model_config_path, cfg.fixed_8b_model_name)

    frozen = FrozenBudgets(
        run_id=cfg.run_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        calibration_split_manifest=str(cfg.manifest_path),
        calibration_split_sha256=_sha256_of_file(cfg.manifest_path),
        calibration_config_path=str(args.config),
        matched_cost_tolerance=cfg.matched_cost_tolerance,
        b_low=FrozenBudget(
            name="B_low",
            max_tiles=selected_adaptive.low_max_tiles,
            model_name=adaptive_model.name,
            adapter_kind=adaptive_model.adapter_kind,
        ),
        b_high=FrozenBudget(
            name="B_high",
            max_tiles=selected_adaptive.high_max_tiles,
            model_name=adaptive_model.name,
            adapter_kind=adaptive_model.adapter_kind,
        ),
        b_fix_2b=FrozenBudget(
            name="B_fix_2B",
            max_tiles=selected_fixed_2b.max_tiles,
            model_name=fixed_2b_model.name,
            adapter_kind=fixed_2b_model.adapter_kind,
        ),
        b_fix_8b=FrozenBudget(
            name="B_fix_8B",
            max_tiles=selected_fixed_8b.max_tiles,
            model_name=fixed_8b_model.name,
            adapter_kind=fixed_8b_model.adapter_kind,
        ),
        adaptive_selection=FrozenAdaptiveSelection(
            target_cost_tiles=selected_adaptive.target_cost_tiles,
            measured_cost_tiles=selected_adaptive.measured_cost_tiles,
            low_max_tiles=selected_adaptive.low_max_tiles,
            high_max_tiles=selected_adaptive.high_max_tiles,
        ),
        fixed_2b_selection=_fixed_selection(selected_fixed_2b),
        fixed_8b_selection=_fixed_selection(selected_fixed_8b),
    )
    frozen_path = write_frozen_budgets(cfg.frozen_budgets_path, frozen)

    _print_summary(
        cfg=cfg,
        selected_adaptive=selected_adaptive,
        selected_fixed_2b=selected_fixed_2b,
        selected_fixed_8b=selected_fixed_8b,
        frozen_path=frozen_path,
        sweep_summaries_path=cfg.sweep_summaries_path,
    )


def _fixed_selection(sel: SelectedFixed) -> FrozenFixedSelection:
    return FrozenFixedSelection(
        target_cost_tiles=sel.target_cost_tiles,
        measured_cost_tiles=sel.measured_cost_tiles,
        max_tiles=sel.max_tiles,
        within_tolerance=sel.within_tolerance,
    )


def _write_sweep_summaries(
    path: Path,
    *,
    adaptive_points: list[AdaptiveSweepPoint],
    fixed_2b_points: list[FixedSweepPoint],
    fixed_8b_points: list[FixedSweepPoint],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in adaptive_points:
            f.write(json.dumps(_adaptive_summary_record(p), sort_keys=True) + "\n")
        for p in fixed_2b_points:
            f.write(json.dumps(_fixed_summary_record(p), sort_keys=True) + "\n")
        for p in fixed_8b_points:
            f.write(json.dumps(_fixed_summary_record(p), sort_keys=True) + "\n")


def _adaptive_summary_record(p: AdaptiveSweepPoint) -> dict:
    s = p.summary
    return {
        "kind": "adaptive",
        "low_max_tiles": p.low_max_tiles,
        "high_max_tiles": p.high_max_tiles,
        "sample_size": s.sample_size,
        "mean_runtime_ms": s.mean_runtime_ms,
        "p95_runtime_ms": s.p95_runtime_ms,
        "mean_output_tokens": s.mean_output_tokens,
        "mean_tile_budget": s.mean_tile_budget,
        "reparse_rate": s.reparse_rate,
        "cost_tiles": s.cost_tiles,
        "output_dir": str(p.output_dir),
    }


def _fixed_summary_record(p: FixedSweepPoint) -> dict:
    s = p.summary
    return {
        "kind": "fixed",
        "model_name": p.model_name,
        "max_tiles": p.max_tiles,
        "sample_size": s.sample_size,
        "mean_runtime_ms": s.mean_runtime_ms,
        "p95_runtime_ms": s.p95_runtime_ms,
        "mean_output_tokens": s.mean_output_tokens,
        "mean_tile_budget": s.mean_tile_budget,
        "reparse_rate": s.reparse_rate,
        "cost_tiles": s.cost_tiles,
        "output_dir": str(p.output_dir),
    }


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def _print_summary(
    *,
    cfg: CalibrationConfig,
    selected_adaptive: SelectedAdaptive,
    selected_fixed_2b: SelectedFixed,
    selected_fixed_8b: SelectedFixed,
    frozen_path: Path,
    sweep_summaries_path: Path,
) -> None:
    print(f"run_id                : {cfg.run_id}")
    print(f"calibration split     : {cfg.manifest_path}")
    print(
        "adaptive pair         : "
        f"B_low(max_tiles={selected_adaptive.low_max_tiles}), "
        f"B_high(max_tiles={selected_adaptive.high_max_tiles}) | "
        f"measured={selected_adaptive.measured_cost_tiles:.3f} tiles, "
        f"target={selected_adaptive.target_cost_tiles:.3f}"
    )
    print(
        "B_fix_2B              : "
        f"max_tiles={selected_fixed_2b.max_tiles} | "
        f"measured={selected_fixed_2b.measured_cost_tiles:.3f}, "
        f"within_tolerance={selected_fixed_2b.within_tolerance}"
    )
    print(
        "B_fix_8B              : "
        f"max_tiles={selected_fixed_8b.max_tiles} | "
        f"measured={selected_fixed_8b.measured_cost_tiles:.3f}, "
        f"within_tolerance={selected_fixed_8b.within_tolerance}"
    )
    print(f"frozen_budgets_path   : {frozen_path}")
    print(f"sweep_summaries_path  : {sweep_summaries_path}")


if __name__ == "__main__":
    main()
