from __future__ import annotations

from adaptive_inference.verifier.codes import (
    DECISION_PASS,
    DECISION_REPARSE,
    DEGENERATE_TABLE,
    NO_TABLE_FOUND,
    RECTANGULAR_INCONSISTENCY,
    SPAN_EXPANSION_FAILED,
)
from adaptive_inference.verifier.structural import verify_page_tables


def _page(table_html: str) -> str:
    return f"# page\n\n{table_html}\n"


def test_clean_stub_output_passes():
    raw = _page(
        "<table>\n"
        "  <tr><th>column_a</th><th>column_b</th></tr>\n"
        "  <tr><td>x</td><td>y</td></tr>\n"
        "</table>"
    )
    result = verify_page_tables(raw)
    assert result.decision == DECISION_PASS
    assert result.failure_codes == ()
    assert result.predicted_table_count == 1
    assert result.html_parse_ok is True
    assert result.span_normalization_ok is True
    assert len(result.tables) == 1
    assert result.tables[0].rows == 2
    assert result.tables[0].cols == 2
    assert result.tables[0].failure_codes == ()


def test_no_tables_triggers_reparse_no_table_found():
    result = verify_page_tables("# page\n\nprose only\n")
    assert result.decision == DECISION_REPARSE
    assert result.failure_codes == (NO_TABLE_FOUND,)
    assert result.predicted_table_count == 0
    assert result.tables == ()


def test_empty_table_triggers_degenerate():
    result = verify_page_tables(_page("<table></table>"))
    assert result.decision == DECISION_REPARSE
    assert DEGENERATE_TABLE in result.failure_codes


def test_all_whitespace_cells_trigger_degenerate():
    raw = _page("<table><tr><td>   </td><td></td></tr></table>")
    result = verify_page_tables(raw)
    assert result.decision == DECISION_REPARSE
    assert DEGENERATE_TABLE in result.failure_codes


def test_ragged_rows_trigger_rectangular_inconsistency():
    raw = _page(
        "<table>"
        "<tr><td>a</td><td>b</td><td>c</td></tr>"
        "<tr><td>d</td><td>e</td></tr>"
        "</table>"
    )
    result = verify_page_tables(raw)
    assert result.decision == DECISION_REPARSE
    assert RECTANGULAR_INCONSISTENCY in result.failure_codes
    assert result.span_normalization_ok is False


def test_overlapping_span_triggers_span_expansion_failed():
    raw = _page(
        "<table>"
        '<tr><td colspan="2">A</td><td rowspan="3">B</td></tr>'
        '<tr><td>C</td><td colspan="3">D</td></tr>'
        "</table>"
    )
    result = verify_page_tables(raw)
    assert result.decision == DECISION_REPARSE
    assert SPAN_EXPANSION_FAILED in result.failure_codes


def test_invalid_span_value_triggers_span_expansion_failed():
    raw = _page('<table><tr><td colspan="0">x</td></tr></table>')
    result = verify_page_tables(raw)
    assert result.decision == DECISION_REPARSE
    assert SPAN_EXPANSION_FAILED in result.failure_codes


def test_one_broken_one_valid_table_still_reparses_page():
    raw = _page(
        "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>"
        "\n\nsome prose\n\n"
        "<table></table>"
    )
    result = verify_page_tables(raw)
    assert result.decision == DECISION_REPARSE
    assert result.predicted_table_count == 2
    assert DEGENERATE_TABLE in result.failure_codes
    assert result.tables[0].failure_codes == ()
    assert result.tables[1].failure_codes == (DEGENERATE_TABLE,)


def test_page_codes_are_unique_even_across_tables():
    raw = _page(
        "<table></table>\n"
        "<table><tr><td>   </td></tr></table>"
    )
    result = verify_page_tables(raw)
    assert result.failure_codes.count(DEGENERATE_TABLE) == 1


def test_single_cell_table_is_not_degenerate():
    # Per the locked spec: 1x1 tables with non-empty content are NOT
    # treated as degenerate for the MVP. Precision-first — don't reparse
    # something that might be a legitimate small table.
    raw = _page("<table><tr><td>value</td></tr></table>")
    result = verify_page_tables(raw)
    assert result.decision == DECISION_PASS


def test_header_only_table_is_not_degenerate():
    raw = _page("<table><tr><th>h1</th><th>h2</th></tr></table>")
    result = verify_page_tables(raw)
    assert result.decision == DECISION_PASS
