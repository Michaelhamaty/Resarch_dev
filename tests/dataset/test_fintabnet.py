"""Unit tests for the FinTabNet loader.

Synthetic PubTabNet-style entries — no network, no real FinTabNet bytes.
"""

from __future__ import annotations

import pytest

from adaptive_inference.dataset.fintabnet import (
    FinTabNetSchemaError,
    SelectedPage,
    assemble_html_from_tokens,
    select_english_table_pages,
)


def _entry(filename: str, structure: list[str], cells: list[list[str]], split: str = "test"):
    return {
        "filename": filename,
        "split": split,
        "html": {
            "structure": {"tokens": structure},
            "cells": [{"tokens": list(cell)} for cell in cells],
        },
    }


def test_assemble_simple_2x2():
    block = {
        "structure": {
            "tokens": [
                "<thead>", "<tr>", "<td>", "</td>", "<td>", "</td>", "</tr>", "</thead>",
                "<tbody>", "<tr>", "<td>", "</td>", "<td>", "</td>", "</tr>", "</tbody>",
            ]
        },
        "cells": [
            {"tokens": ["A"]}, {"tokens": ["B"]},
            {"tokens": ["1"]}, {"tokens": ["2"]},
        ],
    }
    html = assemble_html_from_tokens(block)
    assert "<td>A</td>" in html
    assert "<td>B</td>" in html
    assert "<td>1</td>" in html
    assert "<td>2</td>" in html


def test_assemble_handles_compact_colspan_triplet():
    block = {
        "structure": {
            "tokens": [
                "<tr>",
                "<td", ' colspan="2"', ">",  # compact triple
                "</td>",
                "</tr>",
            ]
        },
        "cells": [{"tokens": ["X"]}],
    }
    html = assemble_html_from_tokens(block)
    assert '<td colspan="2">X</td>' in html


def test_assemble_handles_premerged_colspan_token():
    block = {
        "structure": {
            "tokens": ['<tr>', '<td colspan="3">', "</td>", "</tr>"]
        },
        "cells": [{"tokens": ["Y"]}],
    }
    html = assemble_html_from_tokens(block)
    assert '<td colspan="3">Y</td>' in html


def test_select_basic_test_split():
    entries = [
        _entry("a.png", ["<tr>", "<td>", "</td>", "</tr>"], [["A"]]),
        _entry("b.png", ["<tr>", "<td>", "</td>", "</tr>"], [["B"]]),
    ]
    pages = select_english_table_pages(entries, limit=10)
    ids = sorted(p.page_id for p in pages)
    assert ids == ["a", "b"]
    assert all(p.language == "en" for p in pages)
    assert all(p.row_count == 1 for p in pages)


def test_select_skips_non_target_splits():
    entries = [
        _entry("train1.png", ["<tr>", "<td>", "</td>", "</tr>"], [["X"]], split="train"),
        _entry("test1.png", ["<tr>", "<td>", "</td>", "</tr>"], [["Y"]], split="test"),
    ]
    pages = select_english_table_pages(entries, limit=10)
    assert [p.page_id for p in pages] == ["test1"]


def test_select_can_request_train_split():
    entries = [
        _entry("train1.png", ["<tr>", "<td>", "</td>", "</tr>"], [["X"]], split="train"),
    ]
    pages = select_english_table_pages(entries, limit=10, splits={"train"})
    assert len(pages) == 1


def test_select_groups_multiple_tables_per_page():
    entries = [
        _entry("p.png", ["<tr>", "<td>", "</td>", "</tr>"], [["A"]]),
        _entry("p.png", ["<tr>", "<td>", "</td>", "</tr>"], [["B"]]),
    ]
    pages = select_english_table_pages(entries, limit=10)
    assert len(pages) == 1
    assert "A" in pages[0].table_html and "B" in pages[0].table_html


def test_select_deterministic_sort_and_limit():
    entries = [
        _entry("c.png", ["<tr>", "<td>", "</td>", "</tr>"], [["C"]]),
        _entry("a.png", ["<tr>", "<td>", "</td>", "</tr>"], [["A"]]),
        _entry("b.png", ["<tr>", "<td>", "</td>", "</tr>"], [["B"]]),
    ]
    pages = select_english_table_pages(entries, limit=2)
    assert [p.page_id for p in pages] == ["a", "b"]


def test_select_drops_empty_min_filter():
    entries = [
        _entry("empty.png", ["<tr>", "<td>", "</td>", "</tr>"], [[""]]),
        _entry("full.png",
               ["<tr>", "<td>", "</td>", "<td>", "</td>", "</tr>"],
               [["A"], ["B"]]),
    ]
    pages = select_english_table_pages(entries, limit=10, min_non_empty_cells=2)
    assert [p.page_id for p in pages] == ["full"]


def test_select_detects_merged_cells():
    entries = [
        _entry(
            "merged.png",
            ["<tr>", "<td", ' colspan="2"', ">", "</td>", "</tr>"],
            [["X"]],
        ),
    ]
    pages = select_english_table_pages(entries, limit=10)
    assert pages[0].has_merged_cells is True


def test_select_rejects_negative_limit():
    with pytest.raises(ValueError):
        select_english_table_pages([], limit=0)
    with pytest.raises(ValueError):
        select_english_table_pages([], limit=-1)


def test_select_rejects_negative_min_filter():
    with pytest.raises(ValueError):
        select_english_table_pages([], limit=1, min_non_empty_cells=-1)


def test_assemble_rejects_missing_structure():
    with pytest.raises(FinTabNetSchemaError):
        assemble_html_from_tokens({"cells": []})


def test_assemble_rejects_bad_structure_shape():
    with pytest.raises(FinTabNetSchemaError):
        assemble_html_from_tokens({"structure": {"tokens": "oops"}, "cells": []})


def test_selected_page_dataclass_shape_matches_omnidocbench():
    """Field set must match so downstream records.json is dataset-agnostic."""
    from adaptive_inference.dataset import omnidocbench

    fields_a = set(SelectedPage.__dataclass_fields__.keys())
    fields_b = set(omnidocbench.SelectedPage.__dataclass_fields__.keys())
    assert fields_a == fields_b
