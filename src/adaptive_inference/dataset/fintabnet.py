"""Pure-Python helpers for selecting an English-table subset of FinTabNet.

FinTabNet (Zheng et al. 2021) ships annotations in the PubTabNet token
format: each table entry has

* ``filename`` — relative image (or PDF page) path.
* ``split`` — one of ``"train"`` / ``"val"`` / ``"test"``.
* ``html.structure.tokens`` — sequence of HTML structure tokens
  (``<thead>``, ``<tr>``, ``<td>``, ``<td colspan="2">``, ``</td>``,
  ``</tr>``, ...).
* ``html.cells`` — list of per-cell token lists, aligned with the
  ``<td...>`` openers in the structure stream. Each cell is
  ``{"tokens": ["I", "B", "M", " ", "C", "o", "r", "p"], ...}``.

This module is pure logic + a small regex layer. The fixture builder
(``scripts/data/build_fintabnet_fixture.py``) does the network I/O and
PDF rendering; this file is unit-testable on synthetic inputs.

A FinTabNet *page* may contain multiple table entries. We group by
``filename`` and concatenate the assembled per-table HTML into one
``table_html`` field so the rest of the pipeline (which is page-keyed,
not table-keyed) sees the same ``SelectedPage`` shape as
``omnidocbench``.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .html_metadata import count_non_empty_cells, table_metadata


class FinTabNetSchemaError(ValueError):
    """Raised when a FinTabNet entry does not match an expected shape."""


@dataclass(frozen=True)
class SelectedPage:
    """One English-table page picked from FinTabNet.

    Shape matches ``adaptive_inference.dataset.omnidocbench.SelectedPage``
    so downstream code (records.json, manifests, splits) is dataset-
    agnostic. FinTabNet is English-only by construction, so ``language``
    is always ``"en"`` here; the field is kept for symmetry.
    """

    page_id: str
    image_filename: str
    language: str
    table_html: str
    row_count: int
    col_count: int
    has_merged_cells: bool
    has_nested_headers: bool


_DEFAULT_SPLITS = frozenset({"test"})  # held-out by FinTabNet's own protocol


def select_english_table_pages(
    entries: Iterable[Mapping[str, Any]],
    *,
    limit: int,
    splits: Iterable[str] = _DEFAULT_SPLITS,
    min_non_empty_cells: int = 0,
) -> list[SelectedPage]:
    """Return up to ``limit`` FinTabNet pages from the requested splits.

    Pages are grouped by ``filename`` (one FinTabNet image == one page,
    possibly with multiple table entries). Order is deterministic:
    sorted by ``filename``.
    """

    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    if min_non_empty_cells < 0:
        raise ValueError(
            f"min_non_empty_cells must be non-negative, got {min_non_empty_cells}"
        )

    splits_set = frozenset(s.lower() for s in splits)

    # Group entries by filename, preserving first-seen order so the
    # selection is reproducible from a sorted-input iterator.
    grouped: "OrderedDict[str, list[Mapping[str, Any]]]" = OrderedDict()
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        split = str(entry.get("split", "")).strip().lower()
        if splits_set and split and split not in splits_set:
            continue
        if splits_set and not split:
            # If the entry omits split, accept it only when caller asked
            # for "every split" (passed splits=set()).
            continue
        filename = entry.get("filename")
        if not isinstance(filename, str) or not filename:
            continue
        grouped.setdefault(filename, []).append(entry)

    candidates: list[SelectedPage] = []
    for filename, page_entries in grouped.items():
        try:
            page = _build_page(filename, page_entries)
        except FinTabNetSchemaError:
            continue
        if page is None:
            continue
        if (
            min_non_empty_cells > 0
            and count_non_empty_cells(page.table_html) < min_non_empty_cells
        ):
            continue
        candidates.append(page)

    candidates.sort(key=lambda p: p.image_filename)
    return candidates[:limit]


def _build_page(
    filename: str, entries: Sequence[Mapping[str, Any]]
) -> SelectedPage | None:
    htmls: list[str] = []
    for entry in entries:
        html_block = entry.get("html")
        if not isinstance(html_block, Mapping):
            raise FinTabNetSchemaError(
                f"entry for {filename!r} missing html block"
            )
        assembled = assemble_html_from_tokens(html_block)
        if assembled.strip():
            htmls.append(assembled.strip())

    if not htmls:
        return None

    combined = "\n".join(_wrap_table(h) for h in htmls)
    rows, cols, merged, nested = table_metadata(combined)
    return SelectedPage(
        page_id=_page_id_from_filename(filename),
        image_filename=filename,
        language="en",
        table_html=combined,
        row_count=rows,
        col_count=cols,
        has_merged_cells=merged,
        has_nested_headers=nested,
    )


_TD_OPEN_RE = re.compile(r"^<td\b[^>]*>$|^<td>$", re.IGNORECASE)


def assemble_html_from_tokens(html_block: Mapping[str, Any]) -> str:
    """Zip PubTabNet-style structure tokens with cell-content tokens.

    ``html_block`` is the ``html`` field of a FinTabNet entry:

        {
          "structure": {"tokens": ["<thead>", "<tr>", "<td>", "</td>", ...]},
          "cells": [{"tokens": ["I", "B", "M"]}, {"tokens": ["1"]}, ...]
        }

    The contract: every ``<td>`` (or ``<td ...>``) opener in
    ``structure.tokens`` consumes the next cell from ``cells`` and emits
    the joined cell tokens immediately after the opener. ``</td>`` etc.
    come from the structure stream itself.

    Tolerates both compact PubTabNet token streams (separate ``<td``,
    ``" colspan=\"2\""``, ``>`` tokens) and pre-merged tokens (single
    ``<td colspan="2">`` token).
    """

    structure = html_block.get("structure")
    cells = html_block.get("cells", [])
    if not isinstance(structure, Mapping):
        raise FinTabNetSchemaError("html.structure missing")
    raw_tokens = structure.get("tokens")
    if not isinstance(raw_tokens, list):
        raise FinTabNetSchemaError("html.structure.tokens must be a list")
    if not isinstance(cells, list):
        raise FinTabNetSchemaError("html.cells must be a list")

    merged_tokens = _merge_compact_td_tokens(list(raw_tokens))

    out: list[str] = []
    cell_idx = 0
    for tok in merged_tokens:
        out.append(tok)
        if _TD_OPEN_RE.match(tok) or tok.lower().startswith("<td "):
            if cell_idx < len(cells):
                cell = cells[cell_idx]
                if isinstance(cell, Mapping):
                    cell_tokens = cell.get("tokens", [])
                    if isinstance(cell_tokens, list):
                        out.append("".join(str(t) for t in cell_tokens))
                cell_idx += 1
    return "".join(out)


def _merge_compact_td_tokens(tokens: list[str]) -> list[str]:
    """Merge compact ``["<td", " colspan=\"2\"", ">"]`` triples into one token.

    PubTabNet's released token streams split ``<td colspan="N">`` into
    three pieces. We merge them so the cell-insertion loop above sees
    one ``<td colspan="2">`` token per opener.
    """

    merged: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if isinstance(tok, str) and tok.lower() == "<td":
            buf = [tok]
            j = i + 1
            while j < len(tokens):
                buf.append(tokens[j])
                if isinstance(tokens[j], str) and ">" in tokens[j]:
                    break
                j += 1
            merged.append("".join(buf))
            i = j + 1
        else:
            merged.append(str(tok))
            i += 1
    return merged


def _wrap_table(inner: str) -> str:
    """Ensure ``inner`` is wrapped in a single ``<table>...</table>`` element.

    FinTabNet token streams omit the outer ``<table>`` tag; the scorer
    expects it.
    """

    stripped = inner.strip()
    lower = stripped.lower()
    if lower.startswith("<table"):
        return stripped
    return f"<table>{stripped}</table>"


def _page_id_from_filename(filename: str) -> str:
    stem = filename.rsplit("/", 1)[-1]
    if "." in stem:
        stem = stem.rsplit(".", 1)[0]
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", stem)
    return safe or "page"
