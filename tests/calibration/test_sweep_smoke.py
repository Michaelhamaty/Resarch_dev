"""End-to-end smoke test: run the calibration sweep + selection on a 1-page fixture.

Uses the deterministic stub adapter over a self-contained fixture so
the test has no dependency on ``data/fixtures`` being present. Verifies:

- sweeps produce the expected number of points (adaptive skips pairs
  with ``high < low``),
- summaries carry non-trivial stats from the real run.log.jsonl,
- selection returns valid sweep points,
- the frozen artifact round-trips via ``load_frozen_budgets``.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from PIL import Image

from adaptive_inference.calibration.artifact import (
    FrozenAdaptiveSelection,
    FrozenBudget,
    FrozenBudgets,
    FrozenFixedSelection,
    load_frozen_budgets,
    write_frozen_budgets,
)
from adaptive_inference.calibration.config import load_calibration_config
from adaptive_inference.calibration.select import (
    select_adaptive_pair,
    select_matched_fixed,
)
from adaptive_inference.calibration.sweep import sweep_adaptive, sweep_fixed


def _write_fixture(tmp_path: Path, page_ids: list[str]) -> Path:
    """Build a self-contained calibration-config YAML + referenced files."""

    image_root = tmp_path / "data"
    (image_root / "images").mkdir(parents=True, exist_ok=True)
    for pid in page_ids:
        Image.new("L", (4, 4), color=0).save(
            image_root / "images" / f"{pid}.png", format="PNG"
        )

    records = tmp_path / "records.json"
    records.write_text(
        json.dumps(
            [
                {
                    "page_id": pid,
                    "image_path": f"images/{pid}.png",
                    "language": "en",
                    "contains_table": True,
                    "is_english_table_page": True,
                }
                for pid in page_ids
            ]
        ),
        encoding="utf-8",
    )

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_kind": "calibration_split",
                "schema_version": 1,
                "generated_with_seed": 1,
                "pinning": {},
                "page_ids": page_ids,
            }
        ),
        encoding="utf-8",
    )

    model_cfg = tmp_path / "models.yaml"
    model_cfg.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "internvl2-2b": {
                        "adapter_kind": "stub",
                        "model_id": "stub/2b",
                        "notes": "",
                    },
                    "internvl2-8b": {
                        "adapter_kind": "stub",
                        "model_id": "stub/8b",
                        "notes": "",
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    prompt_cfg = tmp_path / "prompt.yaml"
    prompt_cfg.write_text(
        yaml.safe_dump(
            {
                "id": "table_parse_v1",
                "version": 1,
                "description": "test",
                "template": "parse tables: {page_id}",
            }
        ),
        encoding="utf-8",
    )

    cal_cfg = tmp_path / "calibration.yaml"
    cal_cfg.write_text(
        yaml.safe_dump(
            {
                "run_id": "smoke_calib",
                "split": {
                    "manifest_path": str(manifest),
                    "records_path": str(records),
                    "image_root": str(image_root),
                },
                "prompt_config_path": str(prompt_cfg),
                "model_config_path": str(model_cfg),
                "candidates": {
                    "low_max_tiles": [2, 4],
                    "high_max_tiles": [8, 12],
                    "fixed_2b_max_tiles": [4, 6],
                    "fixed_8b_max_tiles": [4, 6],
                },
                "target_adaptive_cost_tiles": 4.0,
                "matched_cost_tolerance": 0.10,
                "models": {
                    "adaptive": "internvl2-2b",
                    "fixed_2b": "internvl2-2b",
                    "fixed_8b": "internvl2-8b",
                },
                "output": {
                    "sweep_root": str(tmp_path / "sweep"),
                    "sweep_summaries_path": str(tmp_path / "sweep_summaries.jsonl"),
                    "frozen_budgets_path": str(tmp_path / "frozen_budgets.json"),
                },
            }
        ),
        encoding="utf-8",
    )
    return cal_cfg


def test_sweep_end_to_end_and_artifact_roundtrip(tmp_path: Path) -> None:
    cal_cfg_path = _write_fixture(tmp_path, ["page_0001"])
    cfg = load_calibration_config(cal_cfg_path)

    adaptive_points = sweep_adaptive(cfg)
    # 2 low × 2 high = 4 candidate pairs; none have high < low so none skipped.
    assert len(adaptive_points) == 4
    for p in adaptive_points:
        # The sweep output_dir really exists and holds a run.log.jsonl
        # with one line per page processed (one page in this fixture).
        log = p.output_dir / "run.log.jsonl"
        assert log.exists()
        assert p.summary.sample_size == 1
        # Reparse-rate is well-defined (0 under stub, because its output
        # is structurally clean).
        assert p.summary.reparse_rate == 0.0
        # Cost is just low_max_tiles when reparse_rate is 0.
        assert p.summary.cost_tiles == float(p.low_max_tiles)

    fixed_2b_points = sweep_fixed(
        cfg, model_name=cfg.fixed_2b_model_name, candidates=cfg.fixed_2b_max_tiles
    )
    assert len(fixed_2b_points) == 2
    for p in fixed_2b_points:
        assert (p.output_dir / "run.log.jsonl").exists()
        assert p.summary.sample_size == 1
        assert p.summary.cost_tiles == float(p.max_tiles)

    fixed_8b_points = sweep_fixed(
        cfg, model_name=cfg.fixed_8b_model_name, candidates=cfg.fixed_8b_max_tiles
    )
    assert len(fixed_8b_points) == 2

    sel_adaptive = select_adaptive_pair(
        adaptive_points, target_cost_tiles=cfg.target_adaptive_cost_tiles
    )
    # Target 4.0 → B_low=4 is the closest (cost=4.0); among high∈{8,12} tie-break
    # prefers smaller high.
    assert sel_adaptive.low_max_tiles == 4
    assert sel_adaptive.high_max_tiles == 8

    sel_2b = select_matched_fixed(
        fixed_2b_points,
        target_cost_tiles=sel_adaptive.measured_cost_tiles,
        tolerance=cfg.matched_cost_tolerance,
    )
    assert sel_2b.max_tiles == 4
    assert sel_2b.within_tolerance is True

    sel_8b = select_matched_fixed(
        fixed_8b_points,
        target_cost_tiles=sel_adaptive.measured_cost_tiles,
        tolerance=cfg.matched_cost_tolerance,
    )
    assert sel_8b.max_tiles == 4

    # Artifact round-trips.
    fb = FrozenBudgets(
        run_id=cfg.run_id,
        generated_at="2026-04-23T00:00:00+00:00",
        calibration_split_manifest=str(cfg.manifest_path),
        calibration_split_sha256="x" * 64,
        calibration_config_path=str(cal_cfg_path),
        matched_cost_tolerance=cfg.matched_cost_tolerance,
        b_low=FrozenBudget(
            name="B_low",
            max_tiles=sel_adaptive.low_max_tiles,
            model_name=cfg.adaptive_model_name,
            adapter_kind="stub",
        ),
        b_high=FrozenBudget(
            name="B_high",
            max_tiles=sel_adaptive.high_max_tiles,
            model_name=cfg.adaptive_model_name,
            adapter_kind="stub",
        ),
        b_fix_2b=FrozenBudget(
            name="B_fix_2B",
            max_tiles=sel_2b.max_tiles,
            model_name=cfg.fixed_2b_model_name,
            adapter_kind="stub",
        ),
        b_fix_8b=FrozenBudget(
            name="B_fix_8B",
            max_tiles=sel_8b.max_tiles,
            model_name=cfg.fixed_8b_model_name,
            adapter_kind="stub",
        ),
        adaptive_selection=FrozenAdaptiveSelection(
            target_cost_tiles=sel_adaptive.target_cost_tiles,
            measured_cost_tiles=sel_adaptive.measured_cost_tiles,
            low_max_tiles=sel_adaptive.low_max_tiles,
            high_max_tiles=sel_adaptive.high_max_tiles,
        ),
        fixed_2b_selection=FrozenFixedSelection(
            target_cost_tiles=sel_2b.target_cost_tiles,
            measured_cost_tiles=sel_2b.measured_cost_tiles,
            max_tiles=sel_2b.max_tiles,
            within_tolerance=sel_2b.within_tolerance,
        ),
        fixed_8b_selection=FrozenFixedSelection(
            target_cost_tiles=sel_8b.target_cost_tiles,
            measured_cost_tiles=sel_8b.measured_cost_tiles,
            max_tiles=sel_8b.max_tiles,
            within_tolerance=sel_8b.within_tolerance,
        ),
    )
    out_path = write_frozen_budgets(cfg.frozen_budgets_path, fb)
    loaded = load_frozen_budgets(out_path)
    assert loaded == fb


def test_sweep_adaptive_skips_pairs_where_high_lt_low(tmp_path: Path) -> None:
    cal_cfg_path = _write_fixture(tmp_path, ["page_0001"])
    cfg_raw = yaml.safe_load(cal_cfg_path.read_text(encoding="utf-8"))
    # Add a small high candidate that is < some low candidate.
    cfg_raw["candidates"]["low_max_tiles"] = [2, 8]
    cfg_raw["candidates"]["high_max_tiles"] = [4, 12]
    cal_cfg_path.write_text(yaml.safe_dump(cfg_raw), encoding="utf-8")

    cfg = load_calibration_config(cal_cfg_path)
    points = sweep_adaptive(cfg)
    # Pairs: (2,4), (2,12), (8,12). (8,4) is skipped.
    pairs = sorted((p.low_max_tiles, p.high_max_tiles) for p in points)
    assert pairs == [(2, 4), (2, 12), (8, 12)]
