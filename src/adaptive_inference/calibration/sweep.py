"""Drive the existing runners over candidate budgets on the calibration split.

One function per sweep kind. Each call constructs transient
``AdaptiveRunConfig`` / ``RunConfig`` dataclasses in-memory (no YAML
rewriting) that point at the calibration split and a sweep-scoped
output dir, runs the existing orchestrator, and summarizes the
resulting ``run.log.jsonl``.

This module is the only place in ``calibration/`` that imports the
Phase 2/4 runners. Everything else is pure-Python logic over log
records or config values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config.adaptive_runs import AdaptiveRunConfig
from ..config.budgets import Budget
from ..config.models import ModelConfig, load_model_config
from ..config.prompts import PromptTemplate, load_prompt_template
from ..config.runs import RunConfig
from ..runner.adaptive import run_adaptive
from ..runner.single_pass import run_single_pass
from .config import CalibrationConfig
from .summary import (
    SweepPointSummary,
    summarize_adaptive_log,
    summarize_single_pass_log,
)


SPLIT_NAME = "calibration_split"


@dataclass(frozen=True)
class AdaptiveSweepPoint:
    low_max_tiles: int
    high_max_tiles: int
    summary: SweepPointSummary
    output_dir: Path


@dataclass(frozen=True)
class FixedSweepPoint:
    model_name: str
    max_tiles: int
    summary: SweepPointSummary
    output_dir: Path


def sweep_adaptive(cfg: CalibrationConfig) -> list[AdaptiveSweepPoint]:
    """Run the adaptive pipeline once per (low, high) pair.

    Pairs with ``high < low`` are skipped (Contract 1 sanity: reparse
    at the high budget only makes sense when it's at least as large as
    the first pass).
    """

    model_cfg = load_model_config(cfg.model_config_path, cfg.adaptive_model_name)
    prompt = load_prompt_template(cfg.prompt_config_path)
    adaptive_root = cfg.sweep_root / "adaptive"

    points: list[AdaptiveSweepPoint] = []
    for low in cfg.low_max_tiles:
        for high in cfg.high_max_tiles:
            if high < low:
                continue
            point = _run_adaptive_point(
                low=low,
                high=high,
                model_cfg=model_cfg,
                prompt=prompt,
                cfg=cfg,
                adaptive_root=adaptive_root,
            )
            points.append(point)
    return points


def sweep_fixed(
    cfg: CalibrationConfig, *, model_name: str, candidates: tuple[int, ...]
) -> list[FixedSweepPoint]:
    """Run the single-pass pipeline once per candidate tile count."""

    model_cfg = load_model_config(cfg.model_config_path, model_name)
    prompt = load_prompt_template(cfg.prompt_config_path)
    fixed_root = cfg.sweep_root / f"fixed_{model_name}"

    points: list[FixedSweepPoint] = []
    for max_tiles in candidates:
        point = _run_fixed_point(
            max_tiles=max_tiles,
            model_cfg=model_cfg,
            prompt=prompt,
            cfg=cfg,
            fixed_root=fixed_root,
        )
        points.append(point)
    return points


def _run_adaptive_point(
    *,
    low: int,
    high: int,
    model_cfg: ModelConfig,
    prompt: PromptTemplate,
    cfg: CalibrationConfig,
    adaptive_root: Path,
) -> AdaptiveSweepPoint:
    output_dir = adaptive_root / f"low_{low}_high_{high}"
    run_cfg = AdaptiveRunConfig(
        run_id=f"{cfg.run_id}__adaptive__low_{low}_high_{high}",
        split_name=SPLIT_NAME,
        manifest_path=cfg.manifest_path,
        records_path=cfg.records_path,
        image_root=cfg.image_root,
        model_cfg=model_cfg,
        budget_low=Budget(name=f"low_{low}", max_tiles=low),
        budget_high=Budget(name=f"high_{high}", max_tiles=high),
        prompt=prompt,
        output_dir=output_dir,
    )
    summary_result = run_adaptive(run_cfg)
    summary = summarize_adaptive_log(
        summary_result.log_path, low_max_tiles=low, high_max_tiles=high
    )
    return AdaptiveSweepPoint(
        low_max_tiles=low,
        high_max_tiles=high,
        summary=summary,
        output_dir=output_dir,
    )


def _run_fixed_point(
    *,
    max_tiles: int,
    model_cfg: ModelConfig,
    prompt: PromptTemplate,
    cfg: CalibrationConfig,
    fixed_root: Path,
) -> FixedSweepPoint:
    output_dir = fixed_root / f"tiles_{max_tiles}"
    run_cfg = RunConfig(
        run_id=f"{cfg.run_id}__fixed_{model_cfg.name}__tiles_{max_tiles}",
        split_name=SPLIT_NAME,
        manifest_path=cfg.manifest_path,
        records_path=cfg.records_path,
        image_root=cfg.image_root,
        model_cfg=model_cfg,
        budget=Budget(name=f"tiles_{max_tiles}", max_tiles=max_tiles),
        prompt=prompt,
        output_dir=output_dir,
    )
    summary_result = run_single_pass(run_cfg)
    summary = summarize_single_pass_log(summary_result.log_path, max_tiles=max_tiles)
    return FixedSweepPoint(
        model_name=model_cfg.name,
        max_tiles=max_tiles,
        summary=summary,
        output_dir=output_dir,
    )
