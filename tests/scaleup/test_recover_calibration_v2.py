"""Pure-logic tests for scripts/scaleup/recover_calibration_v2.py.

The GPU-touching parts of the recovery script (loading calibration
configs, walking real sweep dirs, writing the frozen artifact) are
exercised by integration on the VM. These tests cover the small set of
pure helpers — pair-dir discovery, summary construction, completeness
filtering, fixed-tile rounding — so a bug in the recovery path is
caught before we run it on real data.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_recovery_module():
    repo_root = Path(__file__).resolve().parents[2]
    spec_path = repo_root / "scripts" / "scaleup" / "recover_calibration_v2.py"
    spec = importlib.util.spec_from_file_location(
        "recover_calibration_v2", spec_path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["recover_calibration_v2"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def recovery():
    return _load_recovery_module()


def _write_adaptive_log(pair_dir: Path, *, n_pages: int, reparse_rate: float) -> None:
    """Write a synthetic Phase-4 adaptive run.log.jsonl into pair_dir."""

    pair_dir.mkdir(parents=True, exist_ok=True)
    log_path = pair_dir / "run.log.jsonl"
    lines: list[str] = []
    n_reparse = int(round(reparse_rate * n_pages))
    for i in range(n_pages):
        reparsed = i < n_reparse
        record = {
            "page_id": f"page_{i:04d}",
            "reparse_triggered": reparsed,
            "first_pass_output_tokens": 100,
            "reparse_output_tokens": 200 if reparsed else None,
            "total_runtime_ms": 48000.0,
        }
        lines.append(json.dumps(record))
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_discover_adaptive_pair_dirs_finds_well_formed_pairs(recovery, tmp_path):
    root = tmp_path / "sweep"
    (root / "adaptive" / "low_2_high_10").mkdir(parents=True)
    (root / "adaptive" / "low_6_high_16").mkdir(parents=True)
    (root / "adaptive" / "low_4_high_12").mkdir(parents=True)

    pairs = recovery.discover_adaptive_pair_dirs(root)
    assert [(p[0], p[1]) for p in pairs] == [(2, 10), (4, 12), (6, 16)]


def test_discover_adaptive_pair_dirs_ignores_malformed_dirs(recovery, tmp_path):
    root = tmp_path / "sweep"
    (root / "adaptive" / "low_2_high_10").mkdir(parents=True)
    (root / "adaptive" / "garbage").mkdir(parents=True)
    (root / "adaptive" / "low_X_high_Y").mkdir(parents=True)

    pairs = recovery.discover_adaptive_pair_dirs(root)
    assert [(p[0], p[1]) for p in pairs] == [(2, 10)]


def test_discover_adaptive_pair_dirs_raises_when_root_missing(recovery, tmp_path):
    with pytest.raises(FileNotFoundError):
        recovery.discover_adaptive_pair_dirs(tmp_path / "no-such-root")


def test_summarize_pair_returns_none_on_missing_log(recovery, tmp_path):
    pair = tmp_path / "low_2_high_10"
    pair.mkdir()
    assert recovery.summarize_pair(2, 10, pair) is None


def test_summarize_pair_returns_none_on_empty_log(recovery, tmp_path):
    pair = tmp_path / "low_2_high_10"
    pair.mkdir()
    (pair / "run.log.jsonl").write_text("", encoding="utf-8")
    assert recovery.summarize_pair(2, 10, pair) is None


def test_summarize_pair_builds_sweep_point(recovery, tmp_path):
    pair = tmp_path / "low_2_high_10"
    _write_adaptive_log(pair, n_pages=50, reparse_rate=0.4)

    point = recovery.summarize_pair(2, 10, pair)
    assert point is not None
    assert point.low_max_tiles == 2
    assert point.high_max_tiles == 10
    assert point.summary.sample_size == 50
    # cost = low + reparse_rate * high  =>  2 + 0.4 * 10 = 6.0
    assert point.summary.cost_tiles == pytest.approx(6.0, abs=1e-9)
    assert point.summary.reparse_rate == pytest.approx(0.4, abs=1e-9)


def test_filter_complete_drops_incomplete(recovery, tmp_path):
    pair_complete = tmp_path / "low_2_high_10"
    _write_adaptive_log(pair_complete, n_pages=50, reparse_rate=0.4)
    pair_incomplete = tmp_path / "low_4_high_12"
    _write_adaptive_log(pair_incomplete, n_pages=7, reparse_rate=0.4)

    p1 = recovery.summarize_pair(2, 10, pair_complete)
    p2 = recovery.summarize_pair(4, 12, pair_incomplete)
    complete = recovery.filter_complete([p1, p2], min_sample_size=50)
    assert [(p.low_max_tiles, p.high_max_tiles) for p in complete] == [(2, 10)]


def test_filter_complete_keeps_all_when_threshold_zero(recovery, tmp_path):
    pair = tmp_path / "low_2_high_10"
    _write_adaptive_log(pair, n_pages=3, reparse_rate=0.0)
    point = recovery.summarize_pair(2, 10, pair)
    assert recovery.filter_complete([point], min_sample_size=0) == [point]


@pytest.mark.parametrize(
    "measured,expected",
    [
        (9.5, 10),     # half-to-even rounds 9.5 -> 10
        (10.0, 10),    # exact
        (10.4, 10),    # below half rounds down
        (10.6, 11),    # above half rounds up
        (8.5, 8),      # half-to-even rounds 8.5 -> 8
        (1.0, 1),      # min
        (0.4, 1),      # clamped to >=1
    ],
)
def test_deterministic_fixed_tiles(recovery, measured, expected):
    assert recovery.deterministic_fixed_tiles(measured) == expected


def test_deterministic_fixed_tiles_rejects_nonpositive(recovery):
    with pytest.raises(ValueError):
        recovery.deterministic_fixed_tiles(0.0)
    with pytest.raises(ValueError):
        recovery.deterministic_fixed_tiles(-1.0)


def test_sha256_of_file_is_stable(recovery, tmp_path):
    path = tmp_path / "x.json"
    path.write_text("hello\n", encoding="utf-8")
    assert (
        recovery.sha256_of_file(path)
        == "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
    )
