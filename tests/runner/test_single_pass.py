"""Unit tests for the single-pass orchestrator wiring."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from adaptive_inference.config.budgets import Budget
from adaptive_inference.config.models import ModelConfig
from adaptive_inference.config.prompts import PromptTemplate
from adaptive_inference.config.runs import RunConfig
from adaptive_inference.runner.runtime_logger import LOG_FILENAME
from adaptive_inference.runner.single_pass import run_single_pass


def _build_inputs(tmp_path: Path, page_ids: list[str]) -> RunConfig:
    image_root = tmp_path / "data"
    images_dir = image_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for pid in page_ids:
        Image.new("L", (4, 4), color=0).save(images_dir / f"{pid}.png", format="PNG")

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

    return RunConfig(
        run_id="test_run",
        split_name="calibration_split",
        manifest_path=manifest,
        records_path=records,
        image_root=image_root,
        model_cfg=ModelConfig(
            name="internvl2-2b", adapter_kind="stub", model_id="x", notes=""
        ),
        budget=Budget(name="low", max_tiles=4),
        prompt=PromptTemplate(
            id="table_parse_v1", version=1, description="d", template="t"
        ),
        output_dir=tmp_path / "out",
    )


def test_run_single_pass_writes_all_artifacts(tmp_path: Path) -> None:
    cfg = _build_inputs(tmp_path, ["page_0001", "page_0002", "page_0003"])
    summary = run_single_pass(cfg)
    assert summary.pages_processed == 3
    assert len(summary.raw_paths) == 3
    assert len(summary.sidecar_paths) == 3
    assert summary.log_path.exists()
    for p in summary.raw_paths:
        assert p.exists()
        assert "<table>" in p.read_text(encoding="utf-8")
    for p in summary.sidecar_paths:
        assert p.exists()


def test_log_has_one_line_per_page(tmp_path: Path) -> None:
    cfg = _build_inputs(tmp_path, ["page_0001", "page_0002"])
    summary = run_single_pass(cfg)
    lines = summary.log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    pages = {json.loads(line)["page_id"] for line in lines}
    assert pages == {"page_0001", "page_0002"}


def test_rerun_is_idempotent_in_log_line_count(tmp_path: Path) -> None:
    cfg = _build_inputs(tmp_path, ["page_0001"])
    run_single_pass(cfg)
    summary = run_single_pass(cfg)
    lines = summary.log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1  # reset_log cleared the previous run


def test_empty_manifest_still_materializes_log(tmp_path: Path) -> None:
    cfg = _build_inputs(tmp_path, [])
    summary = run_single_pass(cfg)
    assert summary.pages_processed == 0
    assert summary.log_path.exists()
    assert summary.log_path.name == LOG_FILENAME
    assert summary.log_path.read_text(encoding="utf-8") == ""
