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


def _write_sidecar(
    run_dir: Path,
    page_id: str,
    raw_text: str,
    *,
    subdir: str = "",
) -> None:
    """Write a Phase 2 (subdir="") or Phase 4 (subdir="final") sidecar pair.

    raw_output_path is always relative to ``run_dir``, matching what both
    output_writer.py and adaptive_writer.py actually emit.
    """

    base = run_dir / subdir if subdir else run_dir
    raw_dir = base / "raw"
    pages_dir = base / "pages"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / f"{page_id}.md"
    raw_rel = str(raw_path.relative_to(run_dir))
    raw_path.write_text(raw_text, encoding="utf-8")

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


def test_adaptive_layout_uses_final_pages(tmp_path: Path) -> None:
    """Phase 4 adaptive runs put sidecars under final/pages/, not pages/."""

    run_dir = tmp_path / "run"
    _write_sidecar(run_dir, "page_a", GOLD_2X2, subdir="final")
    _write_sidecar(run_dir, "page_b", "no html here", subdir="final")
    # Drop a first_pass/ and reparse/ alongside to mimic the real layout;
    # the scorer must ignore them and only score from final/.
    _write_sidecar(run_dir, "page_a", "ignored", subdir="first_pass")
    _write_sidecar(run_dir, "page_b", "also ignored", subdir="reparse")

    gt_path = tmp_path / "gt.json"
    _write_gt(gt_path, {"page_a": GOLD_2X2, "page_b": GOLD_2X2})

    result = score_run(run_dir, gt_path)

    assert result.pages_total == 2
    by_id = {p.page_id: p for p in result.page_scores}
    assert by_id["page_a"].cell_f1 == pytest.approx(1.0)
    assert by_id["page_b"].pred_parse_error == "no_tables"


def test_subpass_run_dir_resolves_raw_path_from_parent(tmp_path: Path) -> None:
    """Adaptive first/final diagnostic points run_dir at a sub-pass dir.

    ``raw_output_path`` is stored relative to the *system* run dir (e.g.
    ``first_pass/raw/page_a.md``). When the diagnostic passes the sub-pass
    dir (``…/adaptive_2b/first_pass``) as ``run_dir``, the direct join would
    double the prefix (``…/first_pass/first_pass/raw/…``). The scorer must
    fall back to resolving from the parent (system) dir instead.
    """

    system_dir = tmp_path / "adaptive_2b"
    raw_dir = system_dir / "first_pass" / "raw"
    pages_dir = system_dir / "first_pass" / "pages"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    (raw_dir / "page_a.md").write_text(GOLD_2X2, encoding="utf-8")
    sidecar = {
        "page_id": "page_a",
        "model_name": "internvl2-2b",
        "budget_name": "B_low",
        "prompt_id": "table_parse_v1",
        "raw_output_path": "first_pass/raw/page_a.md",  # relative to system_dir
        "status": "ok",
    }
    (pages_dir / "page_a.json").write_text(
        json.dumps(sidecar, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    gt_path = tmp_path / "gt.json"
    _write_gt(gt_path, {"page_a": GOLD_2X2})

    # run_dir is the sub-pass dir, not the system dir.
    result = score_run(system_dir / "first_pass", gt_path)

    assert result.pages_total == 1
    by_id = {p.page_id: p for p in result.page_scores}
    assert by_id["page_a"].cell_f1 == pytest.approx(1.0)
    assert by_id["page_a"].pred_parse_error is None


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
