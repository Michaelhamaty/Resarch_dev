"""Pure-Python helpers for selecting an English-table subset of OmniDocBench.

The OmniDocBench HF dataset ships an annotations JSON (one entry per
page) plus a directory of page images. This module *only* parses the
annotations and decides which pages we want — it does no HTTP. The
downloader (``scripts/data/build_omnidocbench_fixture.py``) hands a
pre-loaded list of annotation dicts to ``select_english_table_pages``
and writes the resulting records/ground-truth files. That separation
keeps the selection logic testable on synthetic inputs without
touching the network.

OmniDocBench's annotation shape is large and version-dependent. We pull
out only the fields we need:

* ``page_info.image_path``  — relative image path inside the dataset.
* ``page_info.language`` or ``page_attribute.language`` — to filter to English.
* ``layout_dets``           — list of regions; each region with
  ``category_type == "table"`` may contain an ``html`` field carrying
  the human-annotated ground-truth HTML for that table.

Anything else is ignored. If the upstream schema renames fields, the
helpers raise a clear ``OmniDocBenchSchemaError`` rather than silently
dropping pages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


class OmniDocBenchSchemaError(ValueError):
    """Raised when an annotation entry does not match an expected shape."""


# Field paths we have seen in OmniDocBench releases. Tried in order;
# the first hit wins. Anything not on this list still gets caught by
# the recursive fallback in ``_find_language``.
_LANGUAGE_KEYS = (
    ("page_info", "page_attribute", "language"),  # current real release
    ("page_info", "language"),                    # earlier flat layout
    ("page_attribute", "language"),               # top-level alternative
)
_IMAGE_KEYS = (
    ("page_info", "image_path"),
    ("image_path",),
)
_TABLE_HTML_KEYS = ("html", "html_str", "table_html", "html_seq")
_TABLE_CATEGORIES = frozenset({"table", "Table", "TABLE", "table_body"})
_ENGLISH_VALUES = frozenset({"english", "en", "eng"})


@dataclass(frozen=True)
class SelectedPage:
    """One English-table page picked from OmniDocBench."""

    page_id: str             # stable id we mint from the image filename
    image_filename: str      # relative path inside the dataset (download target)
    language: str            # normalized to lowercase
    table_html: str          # concatenation of every table block on the page
    row_count: int           # max rows across the page's tables (for HardTableRule)
    col_count: int           # max cols across the page's tables
    has_merged_cells: bool   # any rowspan/colspan > 1
    has_nested_headers: bool # multiple <thead> rows or stacked <th>


def select_english_table_pages(
    annotations: Iterable[Mapping[str, Any]],
    *,
    limit: int,
    min_non_empty_cells: int = 0,
) -> list[SelectedPage]:
    """Return up to ``limit`` English pages that have at least one HTML table.

    The order of returned pages is deterministic: sorted by ``image_filename``.
    Within ``limit``, ties never matter because there are no duplicate filenames.

    ``min_non_empty_cells`` drops pages whose combined gold tables have fewer
    than that many cells with non-whitespace text content. Use this to filter
    out OmniDocBench entries where ``<table>`` is used as a layout primitive
    (slide label strips, fillable survey templates) rather than to encode
    tabular data — those pages confound cell-F1 because the model can capture
    the content correctly without emitting ``<table>``. Default ``0`` is a
    no-op (every page with at least one table is kept).
    """

    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    if min_non_empty_cells < 0:
        raise ValueError(
            f"min_non_empty_cells must be non-negative, got {min_non_empty_cells}"
        )

    candidates: list[SelectedPage] = []
    for entry in annotations:
        try:
            page = _try_select_page(entry)
        except OmniDocBenchSchemaError:
            # Skip pages whose schema we cannot parse, but never silently
            # discard recognised English-table pages.
            continue
        if page is None:
            continue
        if min_non_empty_cells > 0 and _count_non_empty_cells(page.table_html) < min_non_empty_cells:
            continue
        candidates.append(page)

    candidates.sort(key=lambda p: p.image_filename)
    return candidates[:limit]


def _try_select_page(entry: Mapping[str, Any]) -> SelectedPage | None:
    language = _find_language(entry)
    if language is None:
        return None
    if language not in _ENGLISH_VALUES:
        return None

    image_path = _first_present(entry, _IMAGE_KEYS)
    if not image_path:
        raise OmniDocBenchSchemaError(f"page entry missing image_path: keys={list(entry)}")

    layout = entry.get("layout_dets")
    if not isinstance(layout, list):
        return None

    table_htmls: list[str] = []
    for region in layout:
        if not isinstance(region, Mapping):
            continue
        if str(region.get("category_type", "")) not in _TABLE_CATEGORIES:
            continue
        html = _first_html(region)
        if html and html.strip():
            table_htmls.append(html.strip())

    if not table_htmls:
        return None

    combined = "\n".join(table_htmls)
    rows, cols, merged, nested = _table_metadata(combined)
    page_id = _page_id_from_filename(str(image_path))
    return SelectedPage(
        page_id=page_id,
        image_filename=str(image_path),
        language=str(language).strip().lower(),
        table_html=combined,
        row_count=rows,
        col_count=cols,
        has_merged_cells=merged,
        has_nested_headers=nested,
    )


def _first_present(
    entry: Mapping[str, Any],
    paths: tuple[tuple[str, ...], ...],
) -> Any | None:
    for path in paths:
        cur: Any = entry
        ok = True
        for key in path:
            if isinstance(cur, Mapping) and key in cur:
                cur = cur[key]
            else:
                ok = False
                break
        if ok and cur is not None and cur != "":
            return cur
    return None


def _find_language(entry: Mapping[str, Any]) -> str | None:
    """Return the page language as a normalized lowercase string, or None.

    First tries the known paths; if none match, falls back to a recursive
    walk looking for any ``language`` key. The recursive fallback is what
    makes the selector resilient to upstream schema renames.
    """

    val = _first_present(entry, _LANGUAGE_KEYS)
    if val is None:
        val = _walk_for_key(entry, "language")
    if val is None:
        return None
    return str(val).strip().lower()


def _walk_for_key(node: Any, target: str, depth: int = 0) -> Any | None:
    """Depth-first search for ``target`` key. Bounded depth to avoid runaway."""

    if depth > 8:
        return None
    if isinstance(node, Mapping):
        if target in node and node[target] not in (None, ""):
            return node[target]
        for v in node.values():
            found = _walk_for_key(v, target, depth + 1)
            if found is not None:
                return found
    elif isinstance(node, list):
        for v in node:
            found = _walk_for_key(v, target, depth + 1)
            if found is not None:
                return found
    return None


def describe_entry(entry: Mapping[str, Any], *, max_str: int = 80) -> dict:
    """Return a structure-only summary of one entry for diagnostics.

    Strings are truncated; nested dicts keep their key shapes; lists report
    length and the type of their first element. Used by the downloader's
    ``--inspect`` mode so we can see real OmniDocBench shapes without
    drowning the terminal in HTML.
    """

    return _describe(entry, max_str=max_str)


def _describe(node: Any, *, max_str: int) -> Any:
    if isinstance(node, Mapping):
        return {k: _describe(v, max_str=max_str) for k, v in node.items()}
    if isinstance(node, list):
        first = _describe(node[0], max_str=max_str) if node else None
        return {"_list_len": len(node), "_first": first}
    if isinstance(node, str):
        return node if len(node) <= max_str else node[: max_str - 1] + "…"
    return node


def _first_html(region: Mapping[str, Any]) -> str | None:
    for key in _TABLE_HTML_KEYS:
        val = region.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return None


# --------------------------------------------------------------------------- #
# Lightweight HTML metadata derivation                                        #
# --------------------------------------------------------------------------- #


_TR_RE = re.compile(r"<tr\b[^>]*>", re.IGNORECASE)
_TD_RE = re.compile(r"<t[dh]\b[^>]*>", re.IGNORECASE)
_CELL_WITH_BODY_RE = re.compile(
    r"<(t[dh])\b[^>]*>(.*?)</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")
_SPAN_GT1_RE = re.compile(r"\b(?:rowspan|colspan)\s*=\s*['\"]?(\d+)", re.IGNORECASE)
_THEAD_RE = re.compile(r"<thead\b[^>]*>", re.IGNORECASE)
_TR_END_RE = re.compile(r"</tr\s*>", re.IGNORECASE)
_TH_RE = re.compile(r"<th\b[^>]*>", re.IGNORECASE)


def _count_non_empty_cells(html: str) -> int:
    """Count ``<td>``/``<th>`` cells whose body has non-whitespace text.

    Strips inline tags (``<br/>``, ``<b>``, ...) before checking. Used by
    ``select_english_table_pages`` to filter out tables-as-layout pages
    where most cells are empty placeholders. Cheap regex matches the rest
    of this module's parsing approach — full HTML parsing happens later
    in the verifier and scorer.
    """

    count = 0
    for match in _CELL_WITH_BODY_RE.finditer(html):
        body = _TAG_RE.sub("", match.group(2))
        if body.strip():
            count += 1
    return count


def _table_metadata(html: str) -> tuple[int, int, bool, bool]:
    """Return ``(row_count, col_count, has_merged_cells, has_nested_headers)``.

    Approximate counts are fine — the only consumer is ``HardTableRule``,
    which uses them as monotone signals. We deliberately use cheap regex
    rather than parsing the HTML; full parsing happens later in the
    verifier and scorer paths where correctness matters more than speed.
    """

    rows = _TR_RE.findall(html)
    row_count = len(rows)

    # Approximate column count: max number of cells in any single row.
    col_count = 0
    for row_segment in _split_rows(html):
        cells = len(_TD_RE.findall(row_segment))
        if cells > col_count:
            col_count = cells

    has_merged = any(int(m.group(1)) > 1 for m in _SPAN_GT1_RE.finditer(html))
    has_nested = _has_nested_headers(html)
    return row_count, col_count, has_merged, has_nested


def _split_rows(html: str) -> list[str]:
    """Return raw <tr>...</tr> substrings (cheap, regex-based)."""

    rows: list[str] = []
    cursor = 0
    for opener in _TR_RE.finditer(html):
        end_match = _TR_END_RE.search(html, opener.end())
        if not end_match:
            break
        rows.append(html[opener.end() : end_match.start()])
        cursor = end_match.end()
    return rows


def _has_nested_headers(html: str) -> bool:
    """Heuristic: more than one <thead>, OR a <tr> containing >1 <th>."""

    if len(_THEAD_RE.findall(html)) > 1:
        return True
    for row_segment in _split_rows(html):
        if len(_TH_RE.findall(row_segment)) > 1:
            return True
    return False


def _page_id_from_filename(image_path: str) -> str:
    """Mint a stable, file-system-safe page_id from the image filename."""

    stem = image_path.rsplit("/", 1)[-1]
    if "." in stem:
        stem = stem.rsplit(".", 1)[0]
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", stem)
    return safe or "page"
