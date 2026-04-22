"""End-to-end smoke test: run the real Phase 4 adaptive config.

Uses the fixture calibration split and the deterministic stub adapter,
so the smoke path is CPU-only and byte-stable under a fixed ``run_id``.
The stub output is structurally clean, so all pages should PASS and no
reparse should fire — this smoke exercises the PASS branch end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_inference.config.adaptive_runs import load_adaptive_run_config
from adaptive_inference.runner.adaptive import run_adaptive


@pytest.fixture
def smoke_adaptive_config_path(configs_dir: Path) -> Path:
    return configs_dir / "runs" / "smoke_adaptive.yaml"


def test_smoke_adaptive_run_produces_expected_artifacts(
    smoke_adaptive_config_path: Path,
    tmp_path: Path,
    repo_root: Path,
) -> None:
    cfg = load_adaptive_run_config(smoke_adaptive_config_path)
    # Repoint paths at the repo root + an isolated output dir.
    cfg = type(cfg)(
        run_id=cfg.run_id,
        split_name=cfg.split_name,
        manifest_path=repo_root / cfg.manifest_path,
        records_path=repo_root / cfg.records_path,
        image_root=repo_root / cfg.image_root,
        model_cfg=cfg.model_cfg,
        budget_low=cfg.budget_low,
        budget_high=cfg.budget_high,
        prompt=cfg.prompt,
        output_dir=tmp_path / "smoke_adaptive_out",
    )

    summary = run_adaptive(cfg)

    # Fixture calibration split has 5 pages.
    assert summary.pages_processed == 5
    assert summary.log_path.exists()

    lines = summary.log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5

    # Stub output is structurally clean -> every page should PASS.
    for line in lines:
        record = json.loads(line)
        assert record["verifier_decision"] == "PASS"
        assert record["reparse_triggered"] is False
        assert record["final_output_source"] == "first_pass"
        assert record["model_name"] == "internvl2-2b"
        assert record["budget_low"] == "low"
        assert record["budget_high"] == "high"
        assert record["prompt_id"] == "table_parse_v1"

    base = summary.output_dir
    assert (base / "first_pass" / "raw").is_dir()
    assert (base / "final" / "raw").is_dir()
    # No reparse branch triggered -> reparse/ may or may not exist, but
    # must contain no artifacts if it does.
    reparse_raw = base / "reparse" / "raw"
    if reparse_raw.exists():
        assert list(reparse_raw.iterdir()) == []
