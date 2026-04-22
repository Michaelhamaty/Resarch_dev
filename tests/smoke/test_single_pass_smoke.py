"""End-to-end smoke test: run the real smoke config against fixture images."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_inference.config.runs import load_run_config
from adaptive_inference.runner.single_pass import run_single_pass


@pytest.fixture
def smoke_run_config_path(configs_dir: Path) -> Path:
    return configs_dir / "runs" / "smoke_single_pass.yaml"


def test_smoke_run_produces_expected_artifacts(
    smoke_run_config_path: Path,
    tmp_path: Path,
    repo_root: Path,
) -> None:
    # Repoint the run config at an isolated output dir so the test never writes
    # into the shared outputs/ tree.
    cfg = load_run_config(smoke_run_config_path)
    cfg = type(cfg)(
        run_id=cfg.run_id,
        split_name=cfg.split_name,
        manifest_path=repo_root / cfg.manifest_path,
        records_path=repo_root / cfg.records_path,
        image_root=repo_root / cfg.image_root,
        model_cfg=cfg.model_cfg,
        budget=cfg.budget,
        prompt=cfg.prompt,
        output_dir=tmp_path / "smoke_out",
    )

    summary = run_single_pass(cfg)

    # Fixture calibration_split is size 5 (see Phase 1 manifest).
    assert summary.pages_processed == 5
    assert summary.log_path.exists()

    lines = summary.log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5

    # Every raw file has the HTML table block dictated by the prompt contract.
    for raw in summary.raw_paths:
        assert "<table>" in raw.read_text(encoding="utf-8")

    # Every sidecar is valid JSON and references the right model + budget.
    for sidecar in summary.sidecar_paths:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        assert payload["model_name"] == "internvl2-2b"
        assert payload["budget_name"] == "low"
        assert payload["prompt_id"] == "table_parse_v1"
