"""Unit tests for the scale-up v2 stratified split builder."""

from __future__ import annotations

import pytest

from adaptive_inference.dataset.records import PageRecord
from adaptive_inference.dataset.stratified_splits import (
    BucketStats,
    SCALEUP_V2_SEED_PHRASE,
    bucket_of,
    derive_seed,
    stratify_and_split,
)


def _rec(page_id: str, row_count: int, merged: bool = False) -> PageRecord:
    return PageRecord(
        page_id=page_id,
        image_path=f"img/{page_id}.png",
        language="en",
        contains_table=True,
        is_english_table_page=True,
        row_count=row_count,
        col_count=4,
        has_merged_cells=merged,
        has_nested_headers=False,
    )


def _population(n_simple: int, n_complex: int, n_very: int) -> list[PageRecord]:
    out: list[PageRecord] = []
    for i in range(n_simple):
        out.append(_rec(f"s{i:04d}", row_count=3))
    for i in range(n_complex):
        out.append(_rec(f"c{i:04d}", row_count=10))
    for i in range(n_very):
        out.append(_rec(f"v{i:04d}", row_count=20))
    return out


def test_bucket_of_thresholds():
    assert bucket_of(_rec("a", 0)) == "simple"
    assert bucket_of(_rec("a", 5)) == "simple"
    assert bucket_of(_rec("a", 6)) == "complex"
    assert bucket_of(_rec("a", 15)) == "complex"
    assert bucket_of(_rec("a", 16)) == "very_complex"
    assert bucket_of(_rec("a", 4, merged=True)) == "very_complex"


def test_derive_seed_deterministic():
    assert derive_seed("abc") == derive_seed("abc")
    assert derive_seed("abc") != derive_seed("abd")


def test_basic_split_disjoint_and_sized():
    pop = _population(400, 150, 100)  # 650 total, proportions ~62/23/15
    result = stratify_and_split(pop, n_total=200, n_calibration=50)

    calib = set(result.calibration)
    held = set(result.held_out)
    assert len(calib) == 50
    assert len(held) == 150
    assert calib & held == set()
    # Every id is real.
    all_pop_ids = {r.page_id for r in pop}
    assert calib | held <= all_pop_ids


def test_bucket_proportions_preserved_in_sample():
    pop = _population(400, 150, 100)
    result = stratify_and_split(pop, n_total=200, n_calibration=50)

    s = result.bucket_stats
    # Sample proportions should be close to population proportions.
    # 400/650 ~= 0.615 → ~123 of 200; allow ±2.
    assert abs(s["simple"].sampled - 123) <= 2
    assert abs(s["complex"].sampled - 46) <= 2
    assert abs(s["very_complex"].sampled - 31) <= 2
    # Sums match.
    assert sum(b.sampled for b in s.values()) == 200
    assert sum(b.in_calibration for b in s.values()) == 50
    assert sum(b.in_held_out for b in s.values()) == 150


def test_split_calibration_held_out_disjoint_per_bucket():
    pop = _population(400, 150, 100)
    result = stratify_and_split(pop, n_total=200, n_calibration=50)
    for stats in result.bucket_stats.values():
        assert stats.in_calibration + stats.in_held_out == stats.sampled


def test_determinism_same_seed():
    pop = _population(400, 150, 100)
    a = stratify_and_split(pop, n_total=200, n_calibration=50, seed=42)
    b = stratify_and_split(pop, n_total=200, n_calibration=50, seed=42)
    assert a.calibration == b.calibration
    assert a.held_out == b.held_out


def test_different_seed_changes_output():
    pop = _population(400, 150, 100)
    a = stratify_and_split(pop, n_total=200, n_calibration=50, seed=1)
    b = stratify_and_split(pop, n_total=200, n_calibration=50, seed=2)
    # Calibration sets should differ on a population of this size.
    assert set(a.calibration) != set(b.calibration)


def test_input_order_does_not_affect_output():
    pop = _population(400, 150, 100)
    a = stratify_and_split(pop, n_total=200, n_calibration=50, seed=7)
    pop_rev = list(reversed(pop))
    b = stratify_and_split(pop_rev, n_total=200, n_calibration=50, seed=7)
    assert a.calibration == b.calibration
    assert a.held_out == b.held_out


def test_sparse_bucket_caps_allocation():
    # Only 5 very_complex in the population — proportional quota for 200/650
    # would ask for ~30 but cap kicks in.
    pop = _population(400, 150, 5)
    result = stratify_and_split(pop, n_total=200, n_calibration=50)
    assert result.bucket_stats["very_complex"].sampled <= 5
    assert sum(b.sampled for b in result.bucket_stats.values()) == 200


def test_too_few_records_raises():
    pop = _population(10, 10, 10)
    with pytest.raises(ValueError):
        stratify_and_split(pop, n_total=200, n_calibration=50)


def test_invalid_sizes_raises():
    pop = _population(400, 150, 100)
    with pytest.raises(ValueError):
        stratify_and_split(pop, n_total=200, n_calibration=200)
    with pytest.raises(ValueError):
        stratify_and_split(pop, n_total=200, n_calibration=0)


def test_default_seed_phrase_is_stable():
    pop = _population(400, 150, 100)
    a = stratify_and_split(pop, n_total=200, n_calibration=50)
    assert a.seed == derive_seed(SCALEUP_V2_SEED_PHRASE)
    assert a.seed_phrase == SCALEUP_V2_SEED_PHRASE
