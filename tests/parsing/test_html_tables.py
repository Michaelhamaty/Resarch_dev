from __future__ import annotations

from adaptive_inference.parsing.html_tables import extract_table_blocks


def test_returns_empty_tuple_on_empty_input():
    assert extract_table_blocks("") == ()


def test_returns_empty_tuple_when_no_tables():
    markdown = "# heading\n\nJust prose, no tables here.\n"
    assert extract_table_blocks(markdown) == ()


def test_finds_single_table():
    raw = "# page\n\n<table><tr><td>a</td></tr></table>\n"
    blocks = extract_table_blocks(raw)
    assert len(blocks) == 1
    assert blocks[0].startswith("<table>")
    assert blocks[0].endswith("</table>")


def test_finds_multiple_tables():
    raw = (
        "<table><tr><td>1</td></tr></table>\n"
        "some prose between\n"
        "<table><tr><td>2</td></tr></table>\n"
    )
    blocks = extract_table_blocks(raw)
    assert len(blocks) == 2
    assert "1" in blocks[0]
    assert "2" in blocks[1]


def test_tolerates_attributes_on_opening_tag():
    raw = '<table class="data" id="t1"><tr><td>x</td></tr></table>'
    blocks = extract_table_blocks(raw)
    assert len(blocks) == 1
    assert blocks[0] == raw


def test_tolerates_case_insensitive_tags():
    raw = "<TABLE><TR><TD>x</TD></TR></TABLE>"
    blocks = extract_table_blocks(raw)
    assert len(blocks) == 1
    assert blocks[0] == raw


def test_preserves_block_content_verbatim():
    inner = "<tr><th>h</th></tr>\n  <tr><td>body\nwith newline</td></tr>"
    raw = f"<table>\n  {inner}\n</table>"
    blocks = extract_table_blocks(raw)
    assert len(blocks) == 1
    assert inner in blocks[0]


def test_ignores_non_table_html_around_block():
    raw = "<div><p>before</p></div><table><tr><td>x</td></tr></table><p>after</p>"
    blocks = extract_table_blocks(raw)
    assert len(blocks) == 1
    assert blocks[0] == "<table><tr><td>x</td></tr></table>"
