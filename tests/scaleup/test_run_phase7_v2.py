"""Unit tests for the Stage 9 aggregator's pure logic.

End-to-end aggregation needs a finished sweep (GPU), so these cover the
disk-free helpers: comparison derivation, random-seed pooling, per-bucket
CIs, and markdown rendering.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "run_phase7_v2",
    Path(__file__).resolve().parents[2] / "scripts" / "scaleup" / "run_phase7_v2.py",
)
m = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = m  # dataclasses need the module registered
_SPEC.loader.exec_module(m)  # type: ignore[union-attr]


def test_derive_comparisons_full_battery():
    comps = m.derive_comparisons(
        ["fixed_2b_low", "fixed_2b_matched", "adaptive_2b",
         "random_2b_seed0", "random_2b_seed1", "random_2b_seed2"]
    )
    pairs = {(c.a, c.b) for c in comps}
    assert ("adaptive_2b", "fixed_2b_matched") in pairs
    assert ("adaptive_2b", "random_2b_pooled") in pairs
    assert len(comps) == 2


def test_derive_comparisons_partial_sweep_no_raise():
    # Only fixed systems present -> no comparisons, no error.
    assert m.derive_comparisons(["fixed_2b_low", "fixed_2b_matched"]) == []
    # Adaptive but no random -> only the matched comparison.
    comps = m.derive_comparisons(["adaptive_2b", "fixed_2b_matched"])
    assert [(c.a, c.b) for c in comps] == [("adaptive_2b", "fixed_2b_matched")]


def test_pool_random_seeds_averages_present_pages():
    pooled = m.pool_random_seeds({
        "random_2b_seed0": {"p1": 0.4, "p2": 0.6},
        "random_2b_seed1": {"p1": 0.6},  # p2 missing for this seed
        "adaptive_2b": {"p1": 1.0},      # ignored
    })
    assert pooled == {"p1": 0.5, "p2": 0.6}


def test_pool_random_seeds_empty_without_random():
    assert m.pool_random_seeds({"adaptive_2b": {"p1": 1.0}}) == {}


def test_bucketed_cis_partitions_by_difficulty():
    per_page = {"a": 0.5, "b": 0.5, "c": 0.9}
    meta = {
        "a": {"row_count": 2, "has_merged_cells": False},    # simple
        "b": {"row_count": 8, "has_merged_cells": False},    # complex
        "c": {"row_count": 30, "has_merged_cells": False},   # very_complex
    }
    out = m.bucketed_cis(per_page, meta)
    assert set(out) == {"simple", "complex", "very_complex"}
    assert out["simple"]["mean"] == 0.5
    assert out["very_complex"]["mean"] == 0.9
    # Single-point bucket -> zero-width CI.
    assert out["simple"]["lo"] == out["simple"]["hi"] == 0.5


def test_render_markdown_includes_systems_and_comparisons():
    results = {
        "generated_at": "2026-05-28T00:00:00+00:00",
        "git_head": "abc123",
        "datasets": {
            "fintabnet": {
                "status": "ok",
                "n_held_out": 100,
                "systems": {
                    "adaptive_2b": {
                        "n_scored": 100,
                        "cell_f1": {"mean": 0.30, "lo": 0.25, "hi": 0.35, "n": 100},
                        "teds": {"mean": 0.40, "lo": 0.35, "hi": 0.45, "n": 100},
                    },
                    "fixed_8b_matched": {"status": "skipped"},
                },
                "comparisons": [{
                    "label": "adaptive vs fixed-matched",
                    "delta_a_minus_b": -0.05,
                    "p_value": 0.012,
                    "n_pairs": 100,
                }],
            }
        },
    }
    md = m.render_markdown(results)
    assert "## fintabnet" in md
    assert "adaptive_2b" in md
    assert "0.3000" in md
    assert "adaptive vs fixed-matched" in md
    assert "skipped" in md  # placeholder row still rendered


def test_ground_truth_path_derivation(tmp_path):
    spec = m.DatasetSpec(
        name="fintabnet",
        split_manifest_path=tmp_path / "held_out.json",
        records_path=tmp_path / "data" / "fintabnet" / "records.json",
        image_root=tmp_path,
        frozen_budgets_path=tmp_path / "frozen.json",
    )
    assert m.ground_truth_path_for(spec).name == "ground_truth.json"
    assert m.ground_truth_path_for(spec).parent.name == "fintabnet"
