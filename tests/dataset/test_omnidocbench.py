"""Tests for the OmniDocBench page-selection helpers (no network)."""

from __future__ import annotations

from adaptive_inference.dataset.omnidocbench import (
    SelectedPage,
    select_english_table_pages,
)


# --------------------------------------------------------------------------- #
# Helpers — synthetic OmniDocBench-shaped entries                             #
# --------------------------------------------------------------------------- #


def _entry(
    *,
    image: str,
    language: str = "english",
    tables: list[str] | None = None,
    page_info: bool = True,
) -> dict:
    layout: list[dict] = []
    for html in tables or []:
        layout.append({"category_type": "table", "html": html})
    if page_info:
        return {
            "page_info": {"image_path": image, "language": language},
            "layout_dets": layout,
        }
    return {
        "image_path": image,
        "page_attribute": {"language": language},
        "layout_dets": layout,
    }


SIMPLE_TABLE = (
    "<table>"
    "<tr><td>a</td><td>b</td></tr>"
    "<tr><td>c</td><td>d</td></tr>"
    "</table>"
)


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


def test_picks_english_table_page() -> None:
    pages = select_english_table_pages(
        [_entry(image="images/a.jpg", tables=[SIMPLE_TABLE])],
        limit=5,
    )
    assert len(pages) == 1
    p = pages[0]
    assert isinstance(p, SelectedPage)
    assert p.page_id == "a"
    assert p.image_filename == "images/a.jpg"
    assert p.language == "english"
    assert SIMPLE_TABLE in p.table_html


def test_skips_non_english_page() -> None:
    entries = [
        _entry(image="images/a.jpg", language="chinese", tables=[SIMPLE_TABLE]),
        _entry(image="images/b.jpg", language="english", tables=[SIMPLE_TABLE]),
    ]
    pages = select_english_table_pages(entries, limit=5)
    assert [p.image_filename for p in pages] == ["images/b.jpg"]


def test_skips_page_with_no_table() -> None:
    entries = [
        _entry(image="images/a.jpg", tables=[]),
        _entry(image="images/b.jpg", tables=[SIMPLE_TABLE]),
    ]
    pages = select_english_table_pages(entries, limit=5)
    assert [p.image_filename for p in pages] == ["images/b.jpg"]


def test_accepts_alternate_schema_path() -> None:
    """Page with `image_path` + `page_attribute.language` (older schema) works."""

    pages = select_english_table_pages(
        [_entry(image="images/a.jpg", tables=[SIMPLE_TABLE], page_info=False)],
        limit=5,
    )
    assert len(pages) == 1


def test_returns_pages_sorted_by_filename_truncated_to_limit() -> None:
    entries = [
        _entry(image=f"images/p_{i:03d}.jpg", tables=[SIMPLE_TABLE]) for i in [3, 1, 5, 2, 4]
    ]
    pages = select_english_table_pages(entries, limit=3)
    assert [p.image_filename for p in pages] == [
        "images/p_001.jpg",
        "images/p_002.jpg",
        "images/p_003.jpg",
    ]


def test_zero_limit_rejected() -> None:
    try:
        select_english_table_pages([_entry(image="x.jpg", tables=[SIMPLE_TABLE])], limit=0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_table_metadata_counts_rows_and_cols() -> None:
    pages = select_english_table_pages(
        [_entry(image="x.jpg", tables=[SIMPLE_TABLE])], limit=1
    )
    assert pages[0].row_count == 2
    assert pages[0].col_count == 2
    assert pages[0].has_merged_cells is False
    assert pages[0].has_nested_headers is False


def test_table_metadata_detects_merged_cells() -> None:
    merged = (
        "<table>"
        "<tr><td colspan='2'>X</td></tr>"
        "<tr><td>c</td><td>d</td></tr>"
        "</table>"
    )
    pages = select_english_table_pages(
        [_entry(image="x.jpg", tables=[merged])], limit=1
    )
    assert pages[0].has_merged_cells is True


def test_table_metadata_detects_nested_headers() -> None:
    nested = (
        "<table>"
        "<thead><tr><th>A</th><th>B</th></tr></thead>"
        "<tr><td>c</td><td>d</td></tr>"
        "</table>"
    )
    pages = select_english_table_pages(
        [_entry(image="x.jpg", tables=[nested])], limit=1
    )
    assert pages[0].has_nested_headers is True


def test_multiple_tables_concatenated_and_max_dims() -> None:
    big = (
        "<table>"
        "<tr><td>1</td><td>2</td><td>3</td><td>4</td></tr>"
        "<tr><td>1</td><td>2</td><td>3</td><td>4</td></tr>"
        "<tr><td>1</td><td>2</td><td>3</td><td>4</td></tr>"
        "</table>"
    )
    pages = select_english_table_pages(
        [_entry(image="x.jpg", tables=[SIMPLE_TABLE, big])], limit=1
    )
    p = pages[0]
    assert p.row_count == 5  # 2 + 3
    assert p.col_count == 4  # max across both tables
    assert SIMPLE_TABLE in p.table_html and big in p.table_html


def test_page_id_sanitized() -> None:
    pages = select_english_table_pages(
        [_entry(image="images/some weird/file name!.png", tables=[SIMPLE_TABLE])],
        limit=1,
    )
    assert pages[0].page_id == "file_name_"  # spaces and ! → _
