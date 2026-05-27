"""Pure-logic tests for scripts/scaleup/smoke_8b_one_page.py.

The script's GPU section (adapter loading, inference, VRAM measurement,
tokenizer download) is not unit-tested — it requires a real CUDA host
and is exercised by hand in Stage 2 of scale-up v2. These tests cover
the page-selection, summary-formatting, and check-aggregation logic so
mistakes in the gates don't go unnoticed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from adaptive_inference.dataset.records import PageRecord


def _load_smoke_module():
    """Import the script module dynamically (it lives under scripts/)."""

    repo_root = Path(__file__).resolve().parents[2]
    spec_path = repo_root / "scripts" / "scaleup" / "smoke_8b_one_page.py"
    spec = importlib.util.spec_from_file_location("smoke_8b_one_page", spec_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["smoke_8b_one_page"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def smoke():
    return _load_smoke_module()


def _rec(page_id: str, contains_table: bool = True) -> PageRecord:
    return PageRecord(
        page_id=page_id,
        image_path=f"omnidocbench/images/{page_id}.png",
        language="en",
        contains_table=contains_table,
        is_english_table_page=True,
    )


def test_pick_smoke_page_returns_first_table_page_by_sorted_id(smoke):
    records = [_rec("c"), _rec("a"), _rec("b")]
    chosen = smoke.pick_smoke_page(records)
    assert chosen.page_id == "a"


def test_pick_smoke_page_skips_records_without_tables(smoke):
    records = [_rec("a", contains_table=False), _rec("b", contains_table=True)]
    chosen = smoke.pick_smoke_page(records)
    assert chosen.page_id == "b"


def test_pick_smoke_page_restricts_to_allowed_ids(smoke):
    records = [_rec("a"), _rec("b"), _rec("c")]
    chosen = smoke.pick_smoke_page(records, allowed_page_ids=["b", "c"])
    assert chosen.page_id == "b"


def test_pick_smoke_page_uses_explicit_override(smoke):
    records = [_rec("a"), _rec("b"), _rec("c")]
    chosen = smoke.pick_smoke_page(records, explicit_page_id="c")
    assert chosen.page_id == "c"


def test_pick_smoke_page_raises_when_no_candidate(smoke):
    records = [_rec("a", contains_table=False)]
    with pytest.raises(ValueError):
        smoke.pick_smoke_page(records)


def test_pick_smoke_page_raises_on_unknown_explicit_id(smoke):
    records = [_rec("a")]
    with pytest.raises(ValueError):
        smoke.pick_smoke_page(records, explicit_page_id="nope")


def test_output_contains_table_finds_lowercase_tag(smoke):
    assert smoke.output_contains_table("<p>hi</p>\n<table>\n<tr><td>x</td></tr>\n</table>")


def test_output_contains_table_finds_attribute_tag(smoke):
    assert smoke.output_contains_table('<table class="foo"><tr><td>x</td></tr></table>')


def test_output_contains_table_negative(smoke):
    assert not smoke.output_contains_table("no table here, just <p>prose</p>")


def test_read_calibration_page_ids(smoke, tmp_path):
    split_path = tmp_path / "calibration.json"
    split_path.write_text(json.dumps({"page_ids": ["x", "y", "z"]}))
    assert smoke.read_calibration_page_ids(split_path) == ["x", "y", "z"]


def test_read_calibration_page_ids_rejects_missing_list(smoke, tmp_path):
    split_path = tmp_path / "calibration.json"
    split_path.write_text(json.dumps({}))
    with pytest.raises(ValueError):
        smoke.read_calibration_page_ids(split_path)


def _make_checks(smoke, **overrides):
    defaults = dict(
        vram_within_budget=True,
        peak_vram_gb=15.0,
        vram_budget_gb=22.0,
        tokenizers_match=True,
        image_token_id_2b=92546,
        image_token_id_8b=92546,
        deterministic_across_runs=True,
        output_contains_table=True,
    )
    defaults.update(overrides)
    return smoke.SmokeChecks(**defaults)


def test_smoke_checks_all_passed_true_when_everything_ok(smoke):
    assert _make_checks(smoke).all_passed() is True


@pytest.mark.parametrize(
    "override",
    [
        {"vram_within_budget": False},
        {"tokenizers_match": False},
        {"deterministic_across_runs": False},
        {"output_contains_table": False},
    ],
)
def test_smoke_checks_all_passed_false_when_any_fail(smoke, override):
    assert _make_checks(smoke, **override).all_passed() is False


def test_smoke_checks_failures_lists_specific_failure(smoke):
    checks = _make_checks(smoke, deterministic_across_runs=False)
    failures = checks.failures()
    assert len(failures) == 1
    assert "determinism" in failures[0]


def test_smoke_checks_failures_empty_when_all_pass(smoke):
    assert _make_checks(smoke).failures() == []


def test_format_summary_marks_pass_and_includes_proceed_line(smoke):
    text = smoke.format_summary(_make_checks(smoke))
    assert "[PASS]" in text
    assert "ALL PASS" in text
    assert "Proceed to Stage 6" in text


def test_format_summary_marks_fail_and_includes_fallback_hint(smoke):
    text = smoke.format_summary(_make_checks(smoke, vram_within_budget=False))
    assert "[FAIL]" in text
    assert "fallback" in text.lower()
    assert "Limitations" in text
