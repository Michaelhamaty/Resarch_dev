"""Per-page cell-F1 + normalized text similarity scorer.

Given a predicted page (raw model output, possibly markdown with embedded
HTML tables) and a ground-truth page (HTML), produce two scalars in
``[0, 1]``:

* **cell_f1** — F1 over ``(table_idx, row, col, normalized_text)`` triples
  after rowspan/colspan expansion. Reuses the verifier's span expansion
  so structurally-broken predictions score 0 on this metric.
* **text_similarity** — ``1 - normalized_edit_distance`` on the cells'
  flattened text content. Rewards near-misses that cell-F1 punishes
  outright.

Both metrics treat the prediction as opaque: malformed HTML is not
repaired, it is penalized. This is consistent with the project's
no-repair-loop rule.

This module is pure-Python and has no new dependencies (lxml is already
used by the verifier).
"""

from __future__ import annotations

from dataclasses import dataclass

from lxml import etree, html

from ..parsing.html_tables import extract_table_blocks
from ..verifier.spans import (
    HtmlParseError,
    RectangularInconsistencyError,
    SpanExpansionError,
    expand_to_grid,
)


@dataclass(frozen=True)
class ScoreResult:
    """Quality numbers for a single page. All similarities in ``[0, 1]``."""

    cell_precision: float
    cell_recall: float
    cell_f1: float
    text_similarity: float
    gold_cell_count: int
    pred_cell_count: int
    pred_parse_error: str | None  # None if the prediction parsed cleanly


def score_page(pred_text: str, gold_html: str) -> ScoreResult:
    """Score one predicted page against ground-truth HTML.

    The prediction may be markdown with embedded ``<table>`` blocks, as
    produced by the runner; ground truth is expected to be HTML containing
    one or more well-formed ``<table>`` elements. Tables are aligned by
    position (first pred table vs first gold table, etc.).
    """

    gold_grids = _expand_all(gold_html, source="gold")
    if isinstance(gold_grids, str):
        # Bad ground truth is a programmer error, not a model failure.
        raise ValueError(f"ground-truth HTML failed to parse: {gold_grids}")

    pred_grids = _expand_all(pred_text, source="pred")
    pred_error: str | None
    if isinstance(pred_grids, str):
        pred_error = pred_grids
        pred_grids = ()
    else:
        pred_error = None

    gold_cells = _cells_from_grids(gold_grids)
    pred_cells = _cells_from_grids(pred_grids)

    cell_precision, cell_recall, cell_f1 = _prf1(pred_cells, gold_cells)
    text_similarity = _text_similarity(pred_grids, gold_grids)

    return ScoreResult(
        cell_precision=cell_precision,
        cell_recall=cell_recall,
        cell_f1=cell_f1,
        text_similarity=text_similarity,
        gold_cell_count=len(gold_cells),
        pred_cell_count=len(pred_cells),
        pred_parse_error=pred_error,
    )


# --------------------------------------------------------------------------- #
# Internals                                                                   #
# --------------------------------------------------------------------------- #


def _expand_all(raw: str, source: str) -> tuple[tuple[tuple[str, ...], ...], ...] | str:
    """Extract and expand every ``<table>`` block in ``raw``.

    Returns the tuple of grids on success, or an error code string on
    failure. The error code is one of: ``no_tables``, ``html_parse_error``,
    ``span_expansion_failed``, ``not_rectangular``.
    """

    blocks = extract_table_blocks(raw)
    if not blocks:
        return "no_tables"
    grids: list[tuple[tuple[str, ...], ...]] = []
    for block in blocks:
        try:
            # Validate structure via the verifier first — same failure
            # codes the runtime sees — then re-expand with span content
            # repeated into every covered cell so a colspan match is
            # not penalised against an unspanned equivalent.
            expand_to_grid(block)
            grids.append(_expand_filled(block))
        except HtmlParseError:
            return "html_parse_error"
        except SpanExpansionError:
            return "span_expansion_failed"
        except RectangularInconsistencyError:
            return "not_rectangular"
    return tuple(grids)


def _expand_filled(table_html: str) -> tuple[tuple[str, ...], ...]:
    """Like ``expand_to_grid`` but repeats span text at every covered cell.

    Mirrors the verifier's parser without re-implementing structural
    validation: callers must run ``expand_to_grid`` first to surface span
    or rectangularity errors. We only build the filled grid here so the
    scorer can compare span-equivalent grids cell-for-cell.
    """

    root = html.fromstring(table_html)
    table = root if root.tag == "table" else root.find(".//table")
    assert table is not None  # guaranteed by the prior expand_to_grid call

    # Collect rows belonging to this table (skip nested tables).
    rows: list[etree._Element] = []
    for tr in table.iter("tr"):
        parent = tr.getparent()
        while parent is not None and parent.tag != "table":
            parent = parent.getparent()
        if parent is table:
            rows.append(tr)

    occupied: dict[tuple[int, int], str] = {}
    next_free_col: dict[int, int] = {}
    max_row = -1
    for r, tr in enumerate(rows):
        for cell in [c for c in tr if c.tag in ("td", "th")]:
            colspan = int((cell.get("colspan") or "1").strip())
            rowspan = int((cell.get("rowspan") or "1").strip())
            text = (cell.text_content() or "").strip()
            c = next_free_col.get(r, 0)
            while (r, c) in occupied:
                c += 1
            for dr in range(rowspan):
                for dc in range(colspan):
                    occupied[(r + dr, c + dc)] = text
                    if r + dr > max_row:
                        max_row = r + dr
            next_free_col[r] = c + colspan

    if max_row < 0:
        return ()
    width = max((c for (_, c) in occupied), default=-1) + 1
    return tuple(
        tuple(occupied.get((r, c), "") for c in range(width))
        for r in range(max_row + 1)
    )


def _cells_from_grids(
    grids: tuple[tuple[tuple[str, ...], ...], ...],
) -> set[tuple[int, int, int, str]]:
    """Flatten grids into a set of ``(table_idx, row, col, normalized_text)``."""

    out: set[tuple[int, int, int, str]] = set()
    for t_idx, grid in enumerate(grids):
        for r_idx, row in enumerate(grid):
            for c_idx, text in enumerate(row):
                out.add((t_idx, r_idx, c_idx, _normalize(text)))
    return out


def _normalize(text: str) -> str:
    """Whitespace-collapse a cell's text. Case is preserved (numbers/names matter)."""

    return " ".join(text.split())


def _prf1(
    pred: set[tuple[int, int, int, str]],
    gold: set[tuple[int, int, int, str]],
) -> tuple[float, float, float]:
    """Standard set precision / recall / F1.

    Convention for empty inputs: if both are empty, score 1.0 (trivially
    perfect — both sides agree there is nothing to predict). If only one
    is empty, score 0.0.
    """

    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not pred or not gold:
        return 0.0, 0.0, 0.0
    tp = len(pred & gold)
    precision = tp / len(pred)
    recall = tp / len(gold)
    if precision + recall == 0.0:
        return 0.0, 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _text_similarity(
    pred_grids: tuple[tuple[tuple[str, ...], ...], ...],
    gold_grids: tuple[tuple[tuple[str, ...], ...], ...],
) -> float:
    """1 - char-level edit distance / max length, on flattened cell text.

    Cells are joined with ``\t`` within rows and ``\n`` between rows, which
    gives the edit distance some structural signal without paying for
    proper tree edit distance. Empty/empty returns 1.0 by convention.
    """

    pred_flat = _flatten(pred_grids)
    gold_flat = _flatten(gold_grids)
    if not pred_flat and not gold_flat:
        return 1.0
    distance = _levenshtein(pred_flat, gold_flat)
    denom = max(len(pred_flat), len(gold_flat))
    if denom == 0:
        return 1.0
    return 1.0 - (distance / denom)


def _flatten(grids: tuple[tuple[tuple[str, ...], ...], ...]) -> str:
    parts: list[str] = []
    for grid in grids:
        for row in grid:
            parts.append("\t".join(_normalize(cell) for cell in row))
        parts.append("")  # blank line between tables
    return "\n".join(parts).strip()


def _levenshtein(a: str, b: str) -> int:
    """Standard O(len(a) * len(b)) Levenshtein with O(min) memory."""

    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current[j] = min(
                current[j - 1] + 1,       # insertion
                previous[j] + 1,          # deletion
                previous[j - 1] + cost,   # substitution
            )
        previous = current
    return previous[-1]
