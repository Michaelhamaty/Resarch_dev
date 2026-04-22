"""English-table filter and the placeholder hard-table rule.

The MVP universe is English table pages (``is_english_table_page=True``).
The hard-table subset is a parallel dimension — pages whose structure looks
difficult by one of a small set of placeholder signals. Real rules will be
refined once OmniDocBench metadata is inspected; the shape is fixed now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .records import PageRecord


_FLAG_FIELDS = frozenset({"has_merged_cells", "has_nested_headers"})


@dataclass(frozen=True)
class HardTableRule:
    min_row_count: int | None = None
    min_col_count: int | None = None
    require_any_of: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "HardTableRule":
        flags = tuple(data.get("require_any_of", ()) or ())
        unknown = [f for f in flags if f not in _FLAG_FIELDS]
        if unknown:
            raise ValueError(
                f"HardTableRule.require_any_of has unknown flags {unknown}; "
                f"supported: {sorted(_FLAG_FIELDS)}"
            )
        return cls(
            min_row_count=_optional_int(data.get("min_row_count")),
            min_col_count=_optional_int(data.get("min_col_count")),
            require_any_of=flags,
        )

    def matches(self, record: PageRecord) -> bool:
        if self.min_row_count is not None and (record.row_count or 0) >= self.min_row_count:
            return True
        if self.min_col_count is not None and (record.col_count or 0) >= self.min_col_count:
            return True
        for flag in self.require_any_of:
            if getattr(record, flag):
                return True
        return False


def filter_english_table_pages(records: list[PageRecord]) -> list[PageRecord]:
    """Keep only English pages that contain a table.

    Order is preserved from the input list; callers that need stable ordering
    should sort the result themselves (splits.py and manifests.py both do).
    """

    return [r for r in records if r.is_english_table_page and r.contains_table]


def filter_hard_table_pages(
    records: list[PageRecord], rule: HardTableRule
) -> list[PageRecord]:
    return [r for r in records if rule.matches(r)]


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
