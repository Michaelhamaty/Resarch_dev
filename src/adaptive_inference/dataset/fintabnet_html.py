"""HTML-shape ingest for the apoidea/fintabnet-html FinTabNet mirror.

The earlier-style FinTabNet annotations ship PubTabNet structure tokens
(``html.structure.tokens`` + ``html.cells``) and are handled by
``adaptive_inference.dataset.fintabnet``. The apoidea mirror is
different: each parquet row carries a single rendered HTML string
(``html_table``) plus the page image as embedded PNG bytes. This module
turns those rows into the same ``SelectedPage`` shape the rest of the
pipeline already consumes, so freeze / manifests / runner / scorer do
not need to know which mirror the data came from.

Pure-Python and unit-testable: the parquet I/O and PNG materialization
live in ``scripts/data/build_fintabnet_fixture.py``; this module only
selects, filters, and reshapes already-loaded rows.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Sequence

from .fintabnet import SelectedPage
from .html_metadata import count_non_empty_cells, table_metadata


@dataclass(frozen=True)
class HtmlRow:
    """One raw row read from an apoidea/fintabnet-html parquet shard.

    ``split`` mirrors the parquet folder it came from (``en/train`` →
    ``"train"``, ``en/validation`` → ``"validation"``). ``image_bytes``
    is the PNG payload from the parquet ``image.bytes`` field. The
    fixture builder is responsible for surfacing this shape; tests
    construct it directly.
    """

    html_table: str
    image_bytes: bytes
    split: str


def select_html_pages(
    rows: Iterable[HtmlRow],
    *,
    limit: int,
    splits: Iterable[str] = ("validation",),
    min_non_empty_cells: int = 0,
) -> list[SelectedPage]:
    """Return up to ``limit`` deterministically-ordered ``SelectedPage``.

    Selection rules:
      * Keep only rows whose ``split`` is in ``splits``.
      * Drop rows whose ``html_table`` is empty / whitespace.
      * Drop rows with fewer than ``min_non_empty_cells`` filled cells.
      * Page id is ``"fintabnet_" + sha1(html_table)[:12]`` — content-
        addressed and stable across runs, shard orderings, and partial
        re-downloads.
      * Output is sorted by ``page_id`` for reproducibility, then sliced
        to ``limit``.
    """

    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    if min_non_empty_cells < 0:
        raise ValueError(
            f"min_non_empty_cells must be non-negative, got {min_non_empty_cells}"
        )

    splits_set = frozenset(s.lower() for s in splits)
    seen_ids: set[str] = set()
    candidates: list[SelectedPage] = []

    for row in rows:
        split = row.split.strip().lower()
        if splits_set and split not in splits_set:
            continue
        html = row.html_table
        if not isinstance(html, str) or not html.strip():
            continue
        if (
            min_non_empty_cells > 0
            and count_non_empty_cells(html) < min_non_empty_cells
        ):
            continue
        page_id = _page_id_from_html(html)
        if page_id in seen_ids:
            # exact-duplicate html across shards — drop the dup
            continue
        seen_ids.add(page_id)

        rows_, cols_, merged, nested = table_metadata(html)
        candidates.append(
            SelectedPage(
                page_id=page_id,
                # image lives in-parquet; the synthesized filename is
                # the canonical on-disk PNG name the fixture builder
                # will write.
                image_filename=f"{page_id}.png",
                language="en",
                table_html=html,
                row_count=rows_,
                col_count=cols_,
                has_merged_cells=merged,
                has_nested_headers=nested,
            )
        )

    candidates.sort(key=lambda p: p.page_id)
    return candidates[:limit]


def _page_id_from_html(html: str) -> str:
    digest = hashlib.sha1(html.encode("utf-8")).hexdigest()[:12]
    return f"fintabnet_{digest}"


__all__: Sequence[str] = ("HtmlRow", "select_html_pages")
