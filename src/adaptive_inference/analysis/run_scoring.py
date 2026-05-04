"""Standalone scorer over a Phase 2 single-pass run directory.

Walks ``{run_dir}/pages/*.json`` sidecars, reads each page's raw model
output via the sidecar's ``raw_output_path`` field, looks up the gold
HTML in a flat ``{page_id: html_str}`` JSON, and calls the cell-F1
scorer. Aggregates per-page results into macro means and optionally
writes a JSON + markdown summary.

Pure-Python; no GPU, no model load, no Phase 6/7 plumbing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..scoring import ScoreResult, score_page


PAGE_SCORES_JSON = "page_scores.json"
SUMMARY_MD = "summary.md"


@dataclass(frozen=True)
class PageScore:
    page_id: str
    cell_precision: float
    cell_recall: float
    cell_f1: float
    text_similarity: float
    gold_cell_count: int
    pred_cell_count: int
    pred_parse_error: str | None
    gold_missing: bool


@dataclass(frozen=True)
class RunScoringResult:
    run_dir: Path
    ground_truth_path: Path
    page_scores: tuple[PageScore, ...]
    macro_cell_f1: float
    macro_text_similarity: float
    pages_total: int
    pages_with_gold: int
    pages_with_parse_error: int
    output_json_path: Path | None
    output_markdown_path: Path | None


def score_run(
    run_dir: Path,
    ground_truth_path: Path,
    *,
    output_dir: Path | None = None,
) -> RunScoringResult:
    """Score every page in a Phase 2 run directory against ground truth."""

    run_dir = Path(run_dir)
    ground_truth_path = Path(ground_truth_path)

    pages_dir = run_dir / "pages"
    if not pages_dir.is_dir():
        raise FileNotFoundError(
            f"run_dir has no pages/ subdirectory: {pages_dir}"
        )

    ground_truth = _load_ground_truth(ground_truth_path)

    sidecars = sorted(pages_dir.glob("*.json"))
    page_scores = tuple(_score_one(sc, run_dir, ground_truth) for sc in sidecars)

    pages_total = len(page_scores)
    pages_with_gold = sum(1 for p in page_scores if not p.gold_missing)
    pages_with_parse_error = sum(1 for p in page_scores if p.pred_parse_error is not None)

    macro_cell_f1, macro_text_similarity = _macros(page_scores)

    out_json: Path | None = None
    out_md: Path | None = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_json = output_dir / PAGE_SCORES_JSON
        out_md = output_dir / SUMMARY_MD
        _write_json_artifact(
            out_json,
            run_dir=run_dir,
            ground_truth_path=ground_truth_path,
            page_scores=page_scores,
            macro_cell_f1=macro_cell_f1,
            macro_text_similarity=macro_text_similarity,
            pages_total=pages_total,
            pages_with_gold=pages_with_gold,
            pages_with_parse_error=pages_with_parse_error,
        )
        _write_markdown_artifact(
            out_md,
            run_dir=run_dir,
            page_scores=page_scores,
            macro_cell_f1=macro_cell_f1,
            macro_text_similarity=macro_text_similarity,
            pages_total=pages_total,
            pages_with_gold=pages_with_gold,
            pages_with_parse_error=pages_with_parse_error,
        )

    return RunScoringResult(
        run_dir=run_dir,
        ground_truth_path=ground_truth_path,
        page_scores=page_scores,
        macro_cell_f1=macro_cell_f1,
        macro_text_similarity=macro_text_similarity,
        pages_total=pages_total,
        pages_with_gold=pages_with_gold,
        pages_with_parse_error=pages_with_parse_error,
        output_json_path=out_json,
        output_markdown_path=out_md,
    )


def _load_ground_truth(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"ground-truth JSON must be a flat object {{page_id: html_str}}: {path}"
        )
    out: dict[str, str] = {}
    for k, v in raw.items():
        if not isinstance(v, str):
            raise ValueError(
                f"ground-truth value for {k!r} is not a string (got {type(v).__name__})"
            )
        out[str(k)] = v
    return out


def _score_one(
    sidecar_path: Path,
    run_dir: Path,
    ground_truth: dict[str, str],
) -> PageScore:
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    page_id = str(sidecar["page_id"])
    raw_rel = sidecar["raw_output_path"]
    pred_text = (run_dir / raw_rel).read_text(encoding="utf-8")

    gold = ground_truth.get(page_id)
    if gold is None:
        return PageScore(
            page_id=page_id,
            cell_precision=0.0,
            cell_recall=0.0,
            cell_f1=0.0,
            text_similarity=0.0,
            gold_cell_count=0,
            pred_cell_count=0,
            pred_parse_error=None,
            gold_missing=True,
        )

    sr: ScoreResult = score_page(pred_text, gold)
    return PageScore(
        page_id=page_id,
        cell_precision=sr.cell_precision,
        cell_recall=sr.cell_recall,
        cell_f1=sr.cell_f1,
        text_similarity=sr.text_similarity,
        gold_cell_count=sr.gold_cell_count,
        pred_cell_count=sr.pred_cell_count,
        pred_parse_error=sr.pred_parse_error,
        gold_missing=False,
    )


def _macros(page_scores: tuple[PageScore, ...]) -> tuple[float, float]:
    scored = [p for p in page_scores if not p.gold_missing]
    if not scored:
        return 0.0, 0.0
    n = len(scored)
    return (
        sum(p.cell_f1 for p in scored) / n,
        sum(p.text_similarity for p in scored) / n,
    )


def _write_json_artifact(
    path: Path,
    *,
    run_dir: Path,
    ground_truth_path: Path,
    page_scores: tuple[PageScore, ...],
    macro_cell_f1: float,
    macro_text_similarity: float,
    pages_total: int,
    pages_with_gold: int,
    pages_with_parse_error: int,
) -> None:
    payload = {
        "run_dir": str(run_dir),
        "ground_truth_path": str(ground_truth_path),
        "summary": {
            "pages_total": pages_total,
            "pages_with_gold": pages_with_gold,
            "pages_with_parse_error": pages_with_parse_error,
            "macro_cell_f1": macro_cell_f1,
            "macro_text_similarity": macro_text_similarity,
        },
        "pages": [asdict(p) for p in page_scores],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_markdown_artifact(
    path: Path,
    *,
    run_dir: Path,
    page_scores: tuple[PageScore, ...],
    macro_cell_f1: float,
    macro_text_similarity: float,
    pages_total: int,
    pages_with_gold: int,
    pages_with_parse_error: int,
) -> None:
    lines: list[str] = []
    lines.append(f"# Scoring summary — `{run_dir.name}`")
    lines.append("")
    lines.append(f"- pages_total: **{pages_total}**")
    lines.append(f"- pages_with_gold: **{pages_with_gold}**")
    lines.append(f"- pages_with_parse_error: **{pages_with_parse_error}**")
    lines.append(f"- macro_cell_f1: **{macro_cell_f1:.4f}**")
    lines.append(f"- macro_text_similarity: **{macro_text_similarity:.4f}**")
    lines.append("")
    lines.append("| page_id | cell_f1 | text_sim | parse_error | gold |")
    lines.append("|---|---:|---:|---|---|")
    for p in page_scores:
        gold_marker = "missing" if p.gold_missing else "ok"
        err = p.pred_parse_error or ""
        lines.append(
            f"| {p.page_id} | {p.cell_f1:.4f} | {p.text_similarity:.4f} | {err} | {gold_marker} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
