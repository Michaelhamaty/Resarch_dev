"""Unit tests for SweepPointSummary statistics."""

from __future__ import annotations

import json
from pathlib import Path

from adaptive_inference.calibration.summary import (
    summarize_adaptive_log,
    summarize_single_pass_log,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n",
        encoding="utf-8",
    )


def test_summarize_single_pass_basic(tmp_path: Path) -> None:
    log = tmp_path / "run.log.jsonl"
    _write_jsonl(
        log,
        [
            {"runtime_ms": 10.0, "output_token_count": 100},
            {"runtime_ms": 20.0, "output_token_count": 200},
            {"runtime_ms": 30.0, "output_token_count": 300},
        ],
    )
    s = summarize_single_pass_log(log, max_tiles=4)
    assert s.sample_size == 3
    assert s.mean_runtime_ms == 20.0
    assert s.mean_output_tokens == 200.0
    assert s.mean_tile_budget == 4.0
    assert s.cost_tiles == 4.0
    assert s.reparse_rate is None


def test_summarize_single_pass_empty_log(tmp_path: Path) -> None:
    log = tmp_path / "run.log.jsonl"
    log.write_text("", encoding="utf-8")
    s = summarize_single_pass_log(log, max_tiles=8)
    assert s.sample_size == 0
    assert s.mean_runtime_ms == 0.0
    assert s.mean_output_tokens == 0.0
    assert s.cost_tiles == 8.0  # fixed cost does not depend on sample


def test_summarize_adaptive_uses_chosen_pass_tokens(tmp_path: Path) -> None:
    log = tmp_path / "run.log.jsonl"
    _write_jsonl(
        log,
        [
            {
                "total_runtime_ms": 10.0,
                "first_pass_output_tokens": 100,
                "reparse_output_tokens": None,
                "reparse_triggered": False,
            },
            {
                "total_runtime_ms": 30.0,
                "first_pass_output_tokens": 100,
                "reparse_output_tokens": 500,
                "reparse_triggered": True,
            },
        ],
    )
    s = summarize_adaptive_log(log, low_max_tiles=4, high_max_tiles=12)
    assert s.sample_size == 2
    assert s.reparse_rate == 0.5
    # reparse page uses reparse tokens (500), non-reparse uses first-pass (100)
    assert s.mean_output_tokens == 300.0
    # cost = (4 + 4+12) / 2 = 10
    assert s.cost_tiles == 10.0
    assert s.mean_tile_budget == 10.0


def test_p95_linear_interpolation(tmp_path: Path) -> None:
    # Build a log with runtimes [0..100]. Linear-interp p95 = 95.0.
    log = tmp_path / "run.log.jsonl"
    _write_jsonl(
        log,
        [
            {"runtime_ms": float(i), "output_token_count": 1}
            for i in range(101)
        ],
    )
    s = summarize_single_pass_log(log, max_tiles=4)
    assert abs(s.p95_runtime_ms - 95.0) < 1e-6


def test_p95_single_sample_returns_value(tmp_path: Path) -> None:
    log = tmp_path / "run.log.jsonl"
    _write_jsonl(log, [{"runtime_ms": 42.0, "output_token_count": 7}])
    s = summarize_single_pass_log(log, max_tiles=4)
    assert s.p95_runtime_ms == 42.0
