"""Unit tests for Phase 6 system-spec construction and filtering."""

from __future__ import annotations

import pytest

from adaptive_inference.experiment.systems import (
    ADAPTIVE_2B,
    FIXED_2B_LOW,
    FIXED_2B_MATCHED,
    FIXED_8B_MATCHED,
    RANDOM_2B,
    build_all_system_specs,
    expand_random_seeds,
    filter_specs,
)


def test_build_all_system_specs_covers_every_required_family() -> None:
    specs = build_all_system_specs((0, 1, 2))
    families = [s.family for s in specs]
    # Every required family appears at least once.
    for fam in (ADAPTIVE_2B, FIXED_2B_LOW, FIXED_2B_MATCHED, RANDOM_2B, FIXED_8B_MATCHED):
        assert fam in families


def test_random_seeds_produce_unique_system_ids() -> None:
    specs = build_all_system_specs((0, 1, 2))
    random_ids = [s.system_id for s in specs if s.family == RANDOM_2B]
    assert random_ids == ["random_2b_seed0", "random_2b_seed1", "random_2b_seed2"]


def test_expand_random_seeds_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        expand_random_seeds(())


def test_filter_specs_none_returns_everything() -> None:
    specs = build_all_system_specs((0,))
    assert filter_specs(specs, None) == specs


def test_filter_specs_by_family_keeps_all_seeds() -> None:
    specs = build_all_system_specs((0, 1, 2))
    kept = filter_specs(specs, (RANDOM_2B,))
    assert {s.family for s in kept} == {RANDOM_2B}
    assert len(kept) == 3


def test_filter_specs_by_specific_system_id() -> None:
    specs = build_all_system_specs((0, 1, 2))
    kept = filter_specs(specs, ("random_2b_seed1",))
    assert [s.system_id for s in kept] == ["random_2b_seed1"]


def test_filter_specs_unknown_raises() -> None:
    specs = build_all_system_specs((0,))
    with pytest.raises(ValueError, match="Unknown"):
        filter_specs(specs, ("nonexistent",))


def test_8b_is_last_so_gate_happens_after_cheap_systems() -> None:
    specs = build_all_system_specs((0, 1, 2))
    assert specs[-1].family == FIXED_8B_MATCHED
