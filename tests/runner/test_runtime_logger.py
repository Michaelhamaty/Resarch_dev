"""Tests for the append-only JSONL runtime logger."""

from __future__ import annotations

import json
from pathlib import Path

from adaptive_inference.inference.types import InferenceResult
from adaptive_inference.runner.runtime_logger import (
    LOG_FILENAME,
    append_page_log,
    reset_log,
)


def _make_result(page_id: str = "page_0001") -> InferenceResult:
    return InferenceResult(
        page_id=page_id,
        model_name="internvl2-2b",
        budget_name="low",
        prompt_id="table_parse_v1",
        raw_text="hi",
        output_token_count=1,
        runtime_ms=0.5,
        started_at="2026-04-21T00:00:00+00:00",
        finished_at="2026-04-21T00:00:01+00:00",
    )


def test_appends_one_line_per_call(tmp_path: Path) -> None:
    append_page_log(
        _make_result("page_0001"),
        run_id="r1",
        split="calibration_split",
        raw_output_path="raw/page_0001.md",
        output_dir=tmp_path,
    )
    append_page_log(
        _make_result("page_0002"),
        run_id="r1",
        split="calibration_split",
        raw_output_path="raw/page_0002.md",
        output_dir=tmp_path,
    )
    log_path = tmp_path / LOG_FILENAME
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["page_id"] == "page_0001"
    assert first["run_id"] == "r1"
    assert first["split"] == "calibration_split"
    assert first["raw_output_path"] == "raw/page_0001.md"
    assert first["status"] == "ok"


def test_log_record_has_required_fields(tmp_path: Path) -> None:
    append_page_log(
        _make_result(),
        run_id="r1",
        split="calibration_split",
        raw_output_path="raw/page_0001.md",
        output_dir=tmp_path,
    )
    record = json.loads((tmp_path / LOG_FILENAME).read_text().splitlines()[0])
    expected = {
        "run_id",
        "page_id",
        "split",
        "model_name",
        "budget_name",
        "prompt_id",
        "runtime_ms",
        "output_token_count",
        "raw_output_path",
        "started_at",
        "finished_at",
        "status",
    }
    assert set(record.keys()) == expected


def test_reset_log_clears_file(tmp_path: Path) -> None:
    append_page_log(
        _make_result(),
        run_id="r1",
        split="calibration_split",
        raw_output_path="raw/page_0001.md",
        output_dir=tmp_path,
    )
    reset_log(tmp_path)
    assert not (tmp_path / LOG_FILENAME).exists()


def test_reset_log_no_op_when_missing(tmp_path: Path) -> None:
    reset_log(tmp_path)  # must not raise
