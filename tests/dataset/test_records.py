from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_inference.dataset import PageRecord, load_page_records


def test_load_page_records_fixture(sample_pages_path: Path) -> None:
    records = load_page_records(sample_pages_path)
    assert len(records) == 20
    assert all(isinstance(r, PageRecord) for r in records)

    by_id = {r.page_id: r for r in records}
    assert by_id["page_0001"].language == "en"
    assert by_id["page_0001"].is_english_table_page is True
    assert by_id["page_0001"].row_count == 25
    assert by_id["page_0020"].is_english_table_page is False


def test_load_page_records_rejects_duplicates(tmp_path: Path) -> None:
    payload = [
        {
            "page_id": "dup",
            "image_path": "x.png",
            "language": "en",
            "contains_table": True,
            "is_english_table_page": True,
        },
        {
            "page_id": "dup",
            "image_path": "y.png",
            "language": "en",
            "contains_table": True,
            "is_english_table_page": True,
        },
    ]
    path = tmp_path / "records.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_page_records(path)


def test_load_page_records_rejects_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"page_id": "only"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required fields"):
        load_page_records(path)


def test_load_page_records_rejects_non_list(tmp_path: Path) -> None:
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Expected a JSON list"):
        load_page_records(path)
