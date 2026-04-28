"""Loader tests: schema-tolerance for both single-pass and adaptive logs."""

from __future__ import annotations

import json

import pytest

from adaptive_inference.analysis.loaders import (
    load_jsonl,
    load_loaded_systems,
    load_phase6_manifest,
    load_split_page_ids,
)


def test_load_phase6_manifest_round_trip(phase7_fixture):
    manifest = load_phase6_manifest(phase7_fixture.phase6_manifest_path)
    assert manifest.schema_version == 1
    assert manifest.header.run_set_id == "phase7_test_run"
    assert manifest.header.prompt_id == "table_parse_v1"
    assert manifest.header.calibration_reparse_rate_degenerate is True
    assert manifest.header.random_seeds == (0, 1)
    assert {e.system_id for e in manifest.entries} == {
        "adaptive_2b",
        "fixed_2b_low",
        "fixed_2b_matched",
        "random_2b_seed0",
        "random_2b_seed1",
        "fixed_8b_matched",
    }


def test_load_phase6_manifest_rejects_bad_schema(tmp_path):
    bad = tmp_path / "manifest.json"
    bad.write_text(json.dumps({"schema_version": 99, "header": {}, "entries": []}))
    with pytest.raises(ValueError, match="schema_version=99"):
        load_phase6_manifest(bad)


def test_load_loaded_systems_pairs_log_records(phase7_fixture):
    manifest = load_phase6_manifest(phase7_fixture.phase6_manifest_path)
    systems = load_loaded_systems(manifest, repo_root=phase7_fixture.root)
    assert len(systems) == len(manifest.entries)
    by_id = {s.entry.system_id: s for s in systems}
    # adaptive_2b log lines have the adaptive schema
    rec = by_id["adaptive_2b"].records[0]
    assert "total_runtime_ms" in rec
    assert rec["reparse_triggered"] is False
    # fixed_2b_low log lines have the single-pass schema
    rec = by_id["fixed_2b_low"].records[0]
    assert "runtime_ms" in rec
    assert "reparse_triggered" not in rec


def test_load_jsonl_skips_blank_lines_and_rejects_non_object(tmp_path):
    p = tmp_path / "f.jsonl"
    p.write_text('{"a":1}\n\n{"b":2}\n', encoding="utf-8")
    assert load_jsonl(p) == ({"a": 1}, {"b": 2})

    bad = tmp_path / "bad.jsonl"
    bad.write_text("[1,2,3]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_jsonl(bad)


def test_load_split_page_ids(phase7_fixture):
    pages = load_split_page_ids(phase7_fixture.held_out_split_path)
    assert pages == phase7_fixture.held_out_page_ids
