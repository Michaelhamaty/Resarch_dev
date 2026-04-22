"""Expand a single HTML ``<table>`` block into a canonical rectangular grid.

This is where the verifier's structural thesis lives: a real table
must survive ``rowspan``/``colspan`` expansion into a rectangle with no
overlaps, no gaps, and no ragged rows. Failures here are not recovered —
they are raised as typed exceptions that the structural driver turns
into stable failure codes.

The grid holds each cell's stripped text content at its top-left
position, with span-filled positions holding the empty string. The
verifier only uses shape + emptiness, but keeping the text content makes
the result useful for Phase 7 analysis without a second parse.
"""

from __future__ import annotations

from lxml import etree, html


class HtmlParseError(Exception):
    """Raised when an extracted ``<table>`` block cannot be parsed."""


class SpanExpansionError(Exception):
    """Raised for invalid rowspan/colspan or overlapping cell placement."""


class RectangularInconsistencyError(Exception):
    """Raised when the expanded grid is not a full rectangle."""


Grid = tuple[tuple[str, ...], ...]


def expand_to_grid(table_html: str) -> Grid:
    """Parse one ``<table>`` block and return its canonical rectangular grid.

    Raises ``HtmlParseError``, ``SpanExpansionError``, or
    ``RectangularInconsistencyError`` on structural problems.
    """

    table = _parse_table(table_html)
    rows = _collect_rows(table)

    occupied: dict[tuple[int, int], str] = {}
    next_free_col: dict[int, int] = {}
    max_row = -1

    for r, tr in enumerate(rows):
        for cell in _cells_in_row(tr):
            colspan = _parse_span(cell.get("colspan"), "colspan")
            rowspan = _parse_span(cell.get("rowspan"), "rowspan")

            c = next_free_col.get(r, 0)
            while (r, c) in occupied:
                c += 1

            for dr in range(rowspan):
                for dc in range(colspan):
                    pos = (r + dr, c + dc)
                    if pos in occupied:
                        raise SpanExpansionError(
                            f"cell overlap at row={pos[0]} col={pos[1]}"
                        )
                    occupied[pos] = ""
                    if pos[0] > max_row:
                        max_row = pos[0]

            occupied[(r, c)] = _cell_text(cell)
            next_free_col[r] = c + colspan

    if max_row < 0:
        return ()

    widths = [
        (max((c for (rr, c) in occupied if rr == r), default=-1) + 1)
        for r in range(max_row + 1)
    ]
    if len(set(widths)) > 1:
        raise RectangularInconsistencyError(f"row widths differ: {widths}")

    width = widths[0]
    for r in range(max_row + 1):
        for c in range(width):
            if (r, c) not in occupied:
                raise RectangularInconsistencyError(
                    f"missing cell at row={r} col={c}"
                )

    return tuple(
        tuple(occupied[(r, c)] for c in range(width))
        for r in range(max_row + 1)
    )


def _parse_table(table_html: str) -> etree._Element:
    if not table_html or not table_html.strip():
        raise HtmlParseError("empty table block")
    try:
        root = html.fromstring(table_html)
    except (etree.ParserError, etree.XMLSyntaxError, ValueError) as exc:
        raise HtmlParseError(str(exc)) from exc

    if root.tag == "table":
        return root
    found = root.find(".//table")
    if found is None:
        raise HtmlParseError("no <table> element in parsed block")
    return found


def _collect_rows(table: etree._Element) -> list[etree._Element]:
    """Return ``<tr>`` elements belonging to this table (skipping nested tables)."""

    rows: list[etree._Element] = []
    for tr in table.iter("tr"):
        # Walk up to the nearest enclosing <table>; skip rows that belong
        # to a nested table to match the plan's MVP scope.
        parent = tr.getparent()
        while parent is not None and parent.tag != "table":
            parent = parent.getparent()
        if parent is table:
            rows.append(tr)
    return rows


def _cells_in_row(tr: etree._Element) -> list[etree._Element]:
    return [child for child in tr if child.tag in ("td", "th")]


def _parse_span(raw: str | None, name: str) -> int:
    if raw is None:
        return 1
    try:
        value = int(raw.strip())
    except (TypeError, ValueError) as exc:
        raise SpanExpansionError(f"non-integer {name}={raw!r}") from exc
    if value < 1:
        raise SpanExpansionError(f"non-positive {name}={value}")
    return value


def _cell_text(cell: etree._Element) -> str:
    return (cell.text_content() or "").strip()
