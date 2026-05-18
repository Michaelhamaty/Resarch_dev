"""Per-system summary tests."""

from __future__ import annotations

import json

import pytest

from adaptive_inference.analysis.loaders import (
    ScoringSummary,
    load_loaded_systems,
    load_phase6_manifest,
    load_scoring_summary,
)
from adaptive_inference.analysis.results import summarize_system


def test_adaptive_summary_fields(phase7_fixture):
    manifest = load_phase6_manifest(phase7_fixture.phase6_manifest_path)
    systems = load_loaded_systems(manifest, repo_root=phase7_fixture.root)
    by_id = {s.entry.system_id: s for s in systems}

    res = summarize_system(by_id["adaptive_2b"])
    assert res.runner == "adaptive"
    assert res.sample_size == 2
    assert res.pages_processed == 2
    assert res.reparse_count == 0
    assert res.reparse_rate == 0.0
    assert res.cost_tiles == pytest.approx(6.0)  # B_low only, no reparses
    assert res.verifier_failure_codes == {}
    assert res.predicted_table_count_hist == {1: 2}
    # adaptive runner has no seed
    assert res.seed is None


def test_random_baseline_summary_carries_seed(phase7_fixture):
    manifest = load_phase6_manifest(phase7_fixture.phase6_manifest_path)
    systems = load_loaded_systems(manifest, repo_root=phase7_fixture.root)
    by_id = {s.entry.system_id: s for s in systems}

    res = summarize_system(by_id["random_2b_seed0"])
    assert res.runner == "adaptive_random"
    assert res.seed == 0
    assert res.random_probability == 0.0
    assert res.note and "calibration-measured" in res.note


def test_single_pass_summary_has_no_reparse_concept(phase7_fixture):
    manifest = load_phase6_manifest(phase7_fixture.phase6_manifest_path)
    systems = load_loaded_systems(manifest, repo_root=phase7_fixture.root)
    by_id = {s.entry.system_id: s for s in systems}

    res = summarize_system(by_id["fixed_2b_low"])
    assert res.runner == "single_pass"
    assert res.reparse_rate is None
    assert res.reparse_count is None
    assert res.verifier_failure_codes is None
    assert res.cost_tiles == pytest.approx(6.0)


def test_summarize_system_without_scoring_leaves_accuracy_fields_none(phase7_fixture):
    manifest = load_phase6_manifest(phase7_fixture.phase6_manifest_path)
    systems = load_loaded_systems(manifest, repo_root=phase7_fixture.root)
    by_id = {s.entry.system_id: s for s in systems}

    res = summarize_system(by_id["adaptive_2b"])
    assert res.macro_cell_f1 is None
    assert res.macro_text_similarity is None
    assert res.pages_with_parse_error is None
    assert res.pages_with_gold is None


def test_summarize_system_merges_scoring_summary(phase7_fixture):
    manifest = load_phase6_manifest(phase7_fixture.phase6_manifest_path)
    systems = load_loaded_systems(manifest, repo_root=phase7_fixture.root)
    by_id = {s.entry.system_id: s for s in systems}

    scoring = ScoringSummary(
        system_id="adaptive_2b",
        page_scores_path=phase7_fixture.root / "fake.json",
        macro_cell_f1=0.1272,
        macro_text_similarity=0.2836,
        pages_total=20,
        pages_with_gold=20,
        pages_with_parse_error=6,
    )
    res = summarize_system(by_id["adaptive_2b"], scoring=scoring)
    assert res.macro_cell_f1 == pytest.approx(0.1272)
    assert res.macro_text_similarity == pytest.approx(0.2836)
    assert res.pages_with_parse_error == 6
    assert res.pages_with_gold == 20
    # Non-accuracy fields preserved from the baseline summarizer.
    assert res.runner == "adaptive"
    assert res.cost_tiles == pytest.approx(6.0)


def test_load_scoring_summary_returns_none_when_missing(tmp_path):
    assert load_scoring_summary("adaptive_2b", tmp_path) is None


def test_load_scoring_summary_parses_real_payload(tmp_path):
    target = tmp_path / "score_phase6_adaptive_2b" / "page_scores.json"
    target.parent.mkdir(parents=True)
    target.write_text(
        json.dumps(
            {
                "run_dir": "outputs/runs/phase6_omnidocbench/adaptive_2b",
                "ground_truth_path": "data/omnidocbench/ground_truth.json",
                "summary": {
                    "pages_total": 20,
                    "pages_with_gold": 20,
                    "pages_with_parse_error": 6,
                    "macro_cell_f1": 0.1272,
                    "macro_text_similarity": 0.2836,
                },
                "pages": [],
            }
        ),
        encoding="utf-8",
    )

    summary = load_scoring_summary("adaptive_2b", tmp_path)
    assert summary is not None
    assert summary.macro_cell_f1 == pytest.approx(0.1272)
    assert summary.pages_with_parse_error == 6
    assert summary.pages_total == 20


def test_load_scoring_summary_rejects_malformed_payload(tmp_path):
    target = tmp_path / "score_phase6_adaptive_2b" / "page_scores.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"no_summary_key_here": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="summary"):
        load_scoring_summary("adaptive_2b", tmp_path)


def test_unknown_runner_raises(phase7_fixture):
    """Sanity: a manifest entry with an unknown runner must fail loudly."""

    manifest = load_phase6_manifest(phase7_fixture.phase6_manifest_path)
    systems = load_loaded_systems(manifest, repo_root=phase7_fixture.root)
    sys = systems[0]
    # Mutate the entry into an unknown runner — using object.__setattr__ because
    # ManifestEntry is a frozen dataclass.
    object.__setattr__(sys.entry, "runner", "totally_made_up")
    with pytest.raises(ValueError, match="unknown runner"):
        summarize_system(sys)
