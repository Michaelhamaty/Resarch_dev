"""PageRecord dataclass and JSON loader.

``PageRecord`` is the shared shape the rest of the pipeline consumes. The
Phase 1 loader reads a JSON list of record dicts (fixture today; real
OmniDocBench adapter later) and returns ``list[PageRecord]``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PageRecord:
    page_id: str
    image_path: str
    language: str
    contains_table: bool
    is_english_table_page: bool
    row_count: int | None = None
    col_count: int | None = None
    has_merged_cells: bool = False
    has_nested_headers: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PageRecord":
        missing = [f for f in _REQUIRED_FIELDS if f not in data]
        if missing:
            raise ValueError(
                f"PageRecord missing required fields {missing} "
                f"(got keys={sorted(data.keys())})"
            )
        return cls(
            page_id=str(data["page_id"]),
            image_path=str(data["image_path"]),
            language=str(data["language"]),
            contains_table=bool(data["contains_table"]),
            is_english_table_page=bool(data["is_english_table_page"]),
            row_count=_optional_int(data.get("row_count")),
            col_count=_optional_int(data.get("col_count")),
            has_merged_cells=bool(data.get("has_merged_cells", False)),
            has_nested_headers=bool(data.get("has_nested_headers", False)),
        )


def load_page_records(path: str | Path) -> list[PageRecord]:
    """Load a JSON list of record dicts from ``path``.

    Duplicate ``page_id`` values raise ``ValueError`` — we never want two
    records to claim the same page ID downstream.
    """

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list at {path}, got {type(raw).__name__}")

    records = [PageRecord.from_mapping(item) for item in raw]
    seen: set[str] = set()
    for rec in records:
        if rec.page_id in seen:
            raise ValueError(f"Duplicate page_id in {path}: {rec.page_id}")
        seen.add(rec.page_id)
    return records


_REQUIRED_FIELDS = (
    "page_id",
    "image_path",
    "language",
    "contains_table",
    "is_english_table_page",
)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
