"""Unit tests for the Phase 4 adaptive orchestrator.

Covers both branches of the escalation policy (PASS and REPARSE), the
artifact layout (first_pass/, reparse/, final/), the run log schema,
idempotent re-run, empty-manifest behavior, and the Contract 1
invariant (at most one reparse per page).
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from adaptive_inference.config.adaptive_runs import AdaptiveRunConfig
from adaptive_inference.config.budgets import Budget
from adaptive_inference.config.models import ModelConfig
from adaptive_inference.config.prompts import PromptTemplate
from adaptive_inference.runner import adaptive as adaptive_module
from adaptive_inference.runner.adaptive import run_adaptive
from adaptive_inference.runner.adaptive_logger import LOG_FILENAME
from adaptive_inference.verifier.codes import (
    DECISION_PASS,
    DECISION_REPARSE,
    NO_TABLE_FOUND,
)
from adaptive_inference.verifier.types import VerifierResult


def _pass_result() -> VerifierResult:
    return VerifierResult(
        decision=DECISION_PASS,
        failure_codes=(),
        predicted_table_count=1,
        html_parse_ok=True,
        span_normalization_ok=True,
        tables=(),
    )


def _reparse_result() -> VerifierResult:
    return VerifierResult(
        decision=DECISION_REPARSE,
        failure_codes=(NO_TABLE_FOUND,),
        predicted_table_count=0,
        html_parse_ok=True,
        span_normalization_ok=True,
        tables=(),
    )


def _build_inputs(tmp_path: Path, page_ids: list[str]) -> AdaptiveRunConfig:
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

    return AdaptiveRunConfig(
        run_id="test_adaptive",
        split_name="calibration_split",
        manifest_path=manifest,
        records_path=records,
        image_root=image_root,
        model_cfg=ModelConfig(
            name="internvl2-2b", adapter_kind="stub", model_id="x", notes=""
        ),
        budget_low=Budget(name="low", max_tiles=4),
        budget_high=Budget(name="high", max_tiles=12),
        prompt=PromptTemplate(
            id="table_parse_v1", version=1, description="d", template="t"
        ),
        output_dir=tmp_path / "out",
    )


# ---------- PASS branch: stub output verifies clean ---------------------


def test_pass_branch_skips_reparse(tmp_path: Path, monkeypatch) -> None:
    cfg = _build_inputs(tmp_path, ["page_0001", "page_0002"])
    monkeypatch.setattr(adaptive_module, "verify_page_tables", lambda _raw: _pass_result())

    summary = run_adaptive(cfg)

    assert summary.pages_processed == 2
    assert summary.reparse_count == 0

    base = cfg.output_dir
    for pid in ("page_0001", "page_0002"):
        assert (base / "first_pass" / "raw" / f"{pid}.md").exists()
        assert (base / "first_pass" / "pages" / f"{pid}.json").exists()
        assert not (base / "reparse" / "raw" / f"{pid}.md").exists()
        assert (base / "final" / "raw" / f"{pid}.md").exists()
        assert (base / "final" / "pages" / f"{pid}.json").exists()

    # Final sidecar carries verifier metadata and points at the first_pass.
    sidecar = json.loads(
        (base / "final" / "pages" / "page_0001.json").read_text(encoding="utf-8")
    )
    assert sidecar["final_output_source"] == "first_pass"
    assert sidecar["reparse_triggered"] is False
    assert sidecar["verifier_decision"] == DECISION_PASS
    assert sidecar["verifier_failure_codes"] == []


# ---------- REPARSE branch: verifier forces a reparse -------------------


def test_reparse_branch_writes_reparse_and_final(tmp_path: Path, monkeypatch) -> None:
    cfg = _build_inputs(tmp_path, ["page_0001"])
    monkeypatch.setattr(
        adaptive_module, "verify_page_tables", lambda _raw: _reparse_result()
    )

    summary = run_adaptive(cfg)

    assert summary.pages_processed == 1
    assert summary.reparse_count == 1

    base = cfg.output_dir
    assert (base / "first_pass" / "raw" / "page_0001.md").exists()
    assert (base / "reparse" / "raw" / "page_0001.md").exists()
    assert (base / "reparse" / "pages" / "page_0001.json").exists()
    assert (base / "final" / "raw" / "page_0001.md").exists()

    sidecar = json.loads(
        (base / "final" / "pages" / "page_0001.json").read_text(encoding="utf-8")
    )
    assert sidecar["final_output_source"] == "reparse"
    assert sidecar["reparse_triggered"] is True
    assert sidecar["verifier_decision"] == DECISION_REPARSE
    assert NO_TABLE_FOUND in sidecar["verifier_failure_codes"]
    # Budget on final should match the high budget (the chosen pass).
    assert sidecar["budget_name"] == "high"


# ---------- Log schema ---------------------------------------------------


def test_log_line_schema_pass_case(tmp_path: Path, monkeypatch) -> None:
    cfg = _build_inputs(tmp_path, ["page_0001"])
    monkeypatch.setattr(adaptive_module, "verify_page_tables", lambda _raw: _pass_result())

    summary = run_adaptive(cfg)
    lines = summary.log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    expected_keys = {
        "run_id",
        "split",
        "page_id",
        "model_name",
        "prompt_id",
        "budget_low",
        "budget_high",
        "reparse_triggered",
        "verifier_decision",
        "verifier_failure_codes",
        "predicted_table_count",
        "first_pass_output_tokens",
        "reparse_output_tokens",
        "first_pass_runtime_ms",
        "verifier_runtime_ms",
        "reparse_runtime_ms",
        "total_runtime_ms",
        "first_pass_raw_path",
        "reparse_raw_path",
        "final_raw_path",
        "final_output_source",
        "status",
    }
    assert set(record.keys()) == expected_keys

    assert record["reparse_triggered"] is False
    assert record["reparse_output_tokens"] is None
    assert record["reparse_runtime_ms"] is None
    assert record["reparse_raw_path"] is None
    assert record["final_output_source"] == "first_pass"
    assert record["first_pass_raw_path"] == "first_pass/raw/page_0001.md"
    assert record["final_raw_path"] == "final/raw/page_0001.md"


def test_log_line_schema_reparse_case(tmp_path: Path, monkeypatch) -> None:
    cfg = _build_inputs(tmp_path, ["page_0001"])
    monkeypatch.setattr(
        adaptive_module, "verify_page_tables", lambda _raw: _reparse_result()
    )

    summary = run_adaptive(cfg)
    record = json.loads(
        summary.log_path.read_text(encoding="utf-8").strip().splitlines()[0]
    )
    assert record["reparse_triggered"] is True
    assert record["reparse_output_tokens"] is not None
    assert record["reparse_runtime_ms"] is not None
    assert record["reparse_raw_path"] == "reparse/raw/page_0001.md"
    assert record["final_output_source"] == "reparse"
    assert record["verifier_decision"] == DECISION_REPARSE


# ---------- Idempotent re-run + empty manifest --------------------------


def test_rerun_resets_log(tmp_path: Path, monkeypatch) -> None:
    cfg = _build_inputs(tmp_path, ["page_0001", "page_0002"])
    monkeypatch.setattr(adaptive_module, "verify_page_tables", lambda _raw: _pass_result())

    run_adaptive(cfg)
    summary = run_adaptive(cfg)
    lines = summary.log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_empty_manifest_materializes_log(tmp_path: Path) -> None:
    cfg = _build_inputs(tmp_path, [])
    summary = run_adaptive(cfg)
    assert summary.pages_processed == 0
    assert summary.reparse_count == 0
    assert summary.log_path.exists()
    assert summary.log_path.name == LOG_FILENAME
    assert summary.log_path.read_text(encoding="utf-8") == ""


# ---------- Contract 1: at most one reparse per page --------------------


def test_adapter_called_at_most_twice_per_page(tmp_path: Path, monkeypatch) -> None:
    cfg = _build_inputs(tmp_path, ["page_0001", "page_0002"])
    # Force REPARSE so every page triggers both passes.
    monkeypatch.setattr(
        adaptive_module, "verify_page_tables", lambda _raw: _reparse_result()
    )

    calls: list[tuple[str, str]] = []

    real_build = adaptive_module.build_adapter

    def counting_build_adapter(model_cfg):
        adapter = real_build(model_cfg)
        real_run = adapter.run

        def wrapped_run(*, page_id, image, budget, prompt):
            calls.append((page_id, budget.name))
            return real_run(page_id=page_id, image=image, budget=budget, prompt=prompt)

        adapter.run = wrapped_run  # type: ignore[method-assign]
        return adapter

    monkeypatch.setattr(adaptive_module, "build_adapter", counting_build_adapter)

    run_adaptive(cfg)

    # Exactly two adapter calls per page: one low, one high. Never a third.
    from collections import Counter

    per_page = Counter(page_id for page_id, _ in calls)
    assert per_page == {"page_0001": 2, "page_0002": 2}
    budgets_per_page = {
        pid: sorted(b for p, b in calls if p == pid) for pid in per_page
    }
    assert budgets_per_page == {
        "page_0001": ["high", "low"],
        "page_0002": ["high", "low"],
    }
