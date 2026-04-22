from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_inference.dataset import (
    HardTableRule,
    PageRecord,
    filter_english_table_pages,
    filter_hard_table_pages,
    load_page_records,
)


def _make_record(page_id: str, **overrides: object) -> PageRecord:
    base = dict(
        page_id=page_id,
        image_path=f"{page_id}.png",
        language="en",
        contains_table=True,
        is_english_table_page=True,
        row_count=5,
        col_count=3,
        has_merged_cells=False,
        has_nested_headers=False,
    )
    base.update(overrides)
    return PageRecord(**base)  # type: ignore[arg-type]


def test_english_filter_drops_non_english_and_non_table(sample_pages_path: Path) -> None:
    records = load_page_records(sample_pages_path)
    universe = filter_english_table_pages(records)
    ids = {r.page_id for r in universe}

    # 14 English+table pages in the fixture (page_0001..page_0014).
    assert len(universe) == 14
    assert ids == {f"page_{i:04d}" for i in range(1, 15)}
    # Non-English or non-table pages are excluded.
    for excluded in ("page_0015", "page_0016", "page_0017", "page_0018", "page_0019", "page_0020"):
        assert excluded not in ids


def test_english_filter_is_deterministic(sample_pages_path: Path) -> None:
    records_a = load_page_records(sample_pages_path)
    records_b = load_page_records(sample_pages_path)
    result_a = filter_english_table_pages(records_a)
    result_b = filter_english_table_pages(records_b)
    assert [r.page_id for r in result_a] == [r.page_id for r in result_b]


def test_hard_rule_matches_each_trigger() -> None:
    rule = HardTableRule(
        min_row_count=20,
        min_col_count=6,
        require_any_of=("has_merged_cells", "has_nested_headers"),
    )
    assert rule.matches(_make_record("a", row_count=25)) is True
    assert rule.matches(_make_record("b", col_count=8)) is True
    assert rule.matches(_make_record("c", has_merged_cells=True)) is True
    assert rule.matches(_make_record("d", has_nested_headers=True)) is True
    assert rule.matches(_make_record("e", row_count=5, col_count=3)) is False


def test_filter_hard_table_pages_matches_fixture(sample_pages_path: Path) -> None:
    records = load_page_records(sample_pages_path)
    universe = filter_english_table_pages(records)
    rule = HardTableRule(
        min_row_count=20,
        min_col_count=6,
        require_any_of=("has_merged_cells", "has_nested_headers"),
    )
    hard = filter_hard_table_pages(universe, rule)
    hard_ids = sorted(r.page_id for r in hard)
    # Fixture is authored so pages 0001..0004 each trip a different trigger.
    assert hard_ids == ["page_0001", "page_0002", "page_0003", "page_0004"]


def test_hard_rule_rejects_unknown_flag() -> None:
    with pytest.raises(ValueError, match="unknown flags"):
        HardTableRule.from_mapping({"require_any_of": ["not_a_real_flag"]})


def test_hard_rule_with_no_triggers_matches_nothing() -> None:
    rule = HardTableRule()
    assert rule.matches(_make_record("a", row_count=1000, col_count=1000)) is False
