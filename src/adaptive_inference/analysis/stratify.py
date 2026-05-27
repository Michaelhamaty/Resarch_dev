"""Difficulty-bucket stratification of per-page scores.

Matches the bucketing used by ``stratified_splits``:
* ``simple``       — row_count < 6
* ``complex``      — 6 <= row_count <= 15
* ``very_complex`` — row_count >= 16 OR has_merged_cells

Used by the results-aggregation script to slice the held-out scores
post-hoc; the runner does not know about buckets.
"""

from __future__ import annotations

from typing import Mapping


def stratify_by_difficulty(
    per_page_scores: Mapping[str, float],
    page_metadata: Mapping[str, Mapping[str, object]],
) -> dict[str, list[tuple[str, float]]]:
    """Bucket ``per_page_scores`` by row-count/merged-cell metadata.

    Returns ``{bucket: [(page_id, score), ...]}``. Pages absent from
    ``page_metadata`` are dropped silently (not bucketable). Buckets
    are stable, sorted by page_id within each bucket.
    """

    buckets: dict[str, list[tuple[str, float]]] = {
        "simple": [],
        "complex": [],
        "very_complex": [],
    }
    for page_id, score in per_page_scores.items():
        meta = page_metadata.get(page_id)
        if meta is None:
            continue
        rows = int(meta.get("row_count") or 0)
        merged = bool(meta.get("has_merged_cells", False))
        bucket = _bucket(rows, merged)
        buckets[bucket].append((page_id, float(score)))
    for bucket in buckets.values():
        bucket.sort(key=lambda pair: pair[0])
    return buckets


def _bucket(row_count: int, has_merged: bool) -> str:
    if row_count >= 16 or has_merged:
        return "very_complex"
    if row_count >= 6:
        return "complex"
    return "simple"
