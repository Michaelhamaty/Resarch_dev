"""Per-system summary tests."""

from __future__ import annotations

import pytest

from adaptive_inference.analysis.loaders import (
    load_loaded_systems,
    load_phase6_manifest,
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
