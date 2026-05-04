"""Tests for the standalone Phase 2 scoring runner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from adaptive_inference.analysis.run_scoring import (
    PAGE_SCORES_JSON,
    SUMMARY_MD,
    PageScore,
    RunScoringResult,
    score_run,
)


def _table(rows: list[list[str]]) -> str:
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table>{body}</table>"


GOLD_2X2 = _table([["a", "b"], ["c", "d"]])
GOLD_OTHER = _table([["x", "y"], ["z", "w"]])


def _write_sidecar(run_dir: Path, page_id: str, raw_text: str) -> None:
    raw_dir = run_dir / "raw"
    pages_dir = run_dir / "pages"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    raw_rel = f"raw/{page_id}.md"
    (run_dir / raw_rel).write_text(raw_text, encoding="utf-8")

    sidecar = {
        "page_id": page_id,
        "model_name": "internvl2-2b",
        "budget_name": "B_low",
        "prompt_id": "table_parse_v1",
        "output_token_count": 9,
        "runtime_ms": 0.4,
        "started_at": "2026-04-01T00:00:00+00:00",
        "finished_at": "2026-04-01T00:00:01+00:00",
        "raw_output_path": raw_rel,
        "status": "ok",
    }
    (pages_dir / f"{page_id}.json").write_text(
        json.dumps(sidecar, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _write_gt(path: Path, mapping: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")


def test_happy_path_two_parseable_pages(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_sidecar(run_dir, "page_a", GOLD_2X2)
    _write_sidecar(run_dir, "page_b", GOLD_OTHER)

    gt_path = tmp_path / "gt.json"
    _write_gt(gt_path, {"page_a": GOLD_2X2, "page_b": GOLD_OTHER})

    result = score_run(run_dir, gt_path)

    assert isinstance(result, RunScoringResult)
    assert result.pages_total == 2
    assert result.pages_with_gold == 2
    assert result.pages_with_parse_error == 0
    assert result.macro_cell_f1 == pytest.approx(1.0)
    assert result.macro_text_similarity == pytest.approx(1.0)
    assert all(isinstance(p, PageScore) for p in result.page_scores)
    assert all(p.pred_parse_error is None for p in result.page_scores)
    assert all(not p.gold_missing for p in result.page_scores)


def test_missing_gold_marked_and_excluded_from_macro(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_sidecar(run_dir, "page_a", GOLD_2X2)
    _write_sidecar(run_dir, "page_orphan", GOLD_2X2)

    gt_path = tmp_path / "gt.json"
    _write_gt(gt_path, {"page_a": GOLD_2X2})  # page_orphan missing

    result = score_run(run_dir, gt_path)

    assert result.pages_total == 2
    assert result.pages_with_gold == 1
    by_id = {p.page_id: p for p in result.page_scores}
    assert by_id["page_orphan"].gold_missing is True
    assert by_id["page_a"].gold_missing is False
    # macro is over the one scored page only.
    assert result.macro_cell_f1 == pytest.approx(1.0)


def test_pred_parse_error_counts_in_macro_as_zero(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_sidecar(run_dir, "page_a", GOLD_2X2)
    _write_sidecar(run_dir, "page_no_table", "just prose, no html here")

    gt_path = tmp_path / "gt.json"
    _write_gt(gt_path, {"page_a": GOLD_2X2, "page_no_table": GOLD_2X2})

    result = score_run(run_dir, gt_path)

    assert result.pages_total == 2
    assert result.pages_with_gold == 2
    assert result.pages_with_parse_error == 1
    by_id = {p.page_id: p for p in result.page_scores}
    assert by_id["page_no_table"].pred_parse_error == "no_tables"
    assert by_id["page_no_table"].cell_f1 == 0.0
    # macro = mean(1.0, 0.0) = 0.5
    assert result.macro_cell_f1 == pytest.approx(0.5)


def test_output_artifacts_written(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_sidecar(run_dir, "page_a", GOLD_2X2)
    gt_path = tmp_path / "gt.json"
    _write_gt(gt_path, {"page_a": GOLD_2X2})
    out_dir = tmp_path / "out"

    result = score_run(run_dir, gt_path, output_dir=out_dir)

    assert result.output_json_path == out_dir / PAGE_SCORES_JSON
    assert result.output_markdown_path == out_dir / SUMMARY_MD
    assert result.output_json_path.is_file()
    assert result.output_markdown_path.is_file()

    payload = json.loads(result.output_json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["pages_total"] == 1
    assert payload["summary"]["macro_cell_f1"] == pytest.approx(1.0)
    assert len(payload["pages"]) == 1
    assert payload["pages"][0]["page_id"] == "page_a"

    md = result.output_markdown_path.read_text(encoding="utf-8")
    assert "macro_cell_f1" in md
    assert "page_a" in md


def test_page_scores_sorted_by_page_id(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    # Create in non-sorted order; glob() returns sorted in our impl.
    _write_sidecar(run_dir, "page_z", GOLD_2X2)
    _write_sidecar(run_dir, "page_a", GOLD_2X2)
    _write_sidecar(run_dir, "page_m", GOLD_2X2)
    gt_path = tmp_path / "gt.json"
    _write_gt(gt_path, {"page_a": GOLD_2X2, "page_m": GOLD_2X2, "page_z": GOLD_2X2})

    result = score_run(run_dir, gt_path)

    ids = [p.page_id for p in result.page_scores]
    assert ids == ["page_a", "page_m", "page_z"]


def test_missing_run_dir_raises(tmp_path: Path) -> None:
    gt_path = tmp_path / "gt.json"
    _write_gt(gt_path, {})
    with pytest.raises(FileNotFoundError):
        score_run(tmp_path / "nope", gt_path)


def test_cli_smoke(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_sidecar(run_dir, "page_a", GOLD_2X2)
    gt_path = tmp_path / "gt.json"
    _write_gt(gt_path, {"page_a": GOLD_2X2})
    out_dir = tmp_path / "out"

    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "analysis" / "score_run.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--run-dir",
            str(run_dir),
            "--ground-truth",
            str(gt_path),
            "--output-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "macro_cell_f1" in proc.stdout
    assert (out_dir / PAGE_SCORES_JSON).is_file()
