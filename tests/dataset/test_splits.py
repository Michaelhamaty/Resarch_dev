from __future__ import annotations

import pytest

from adaptive_inference.dataset import split_page_ids


PAGE_IDS = [f"page_{i:04d}" for i in range(1, 15)]


def test_split_is_deterministic_with_same_seed() -> None:
    a = split_page_ids(PAGE_IDS, calibration_size=5, eval_size=None, seed=1337)
    b = split_page_ids(PAGE_IDS, calibration_size=5, eval_size=None, seed=1337)
    assert a == b


def test_split_differs_across_seeds() -> None:
    a = split_page_ids(PAGE_IDS, calibration_size=5, eval_size=None, seed=1)
    b = split_page_ids(PAGE_IDS, calibration_size=5, eval_size=None, seed=2)
    assert a.calibration != b.calibration


def test_split_input_order_does_not_matter() -> None:
    baseline = split_page_ids(PAGE_IDS, calibration_size=5, eval_size=None, seed=1337)
    shuffled_input = list(reversed(PAGE_IDS))
    result = split_page_ids(shuffled_input, calibration_size=5, eval_size=None, seed=1337)
    assert result == baseline


def test_split_sets_are_disjoint() -> None:
    result = split_page_ids(PAGE_IDS, calibration_size=5, eval_size=None, seed=1337)
    cal = set(result.calibration)
    held_out = set(result.held_out_eval)
    assert cal.isdisjoint(held_out)
    assert cal | held_out | set(result.unused) == set(PAGE_IDS)


def test_split_sizes_match_config() -> None:
    result = split_page_ids(PAGE_IDS, calibration_size=5, eval_size=None, seed=1337)
    assert len(result.calibration) == 5
    assert len(result.held_out_eval) == len(PAGE_IDS) - 5
    assert len(result.unused) == 0


def test_split_honors_explicit_eval_size() -> None:
    result = split_page_ids(PAGE_IDS, calibration_size=4, eval_size=6, seed=1337)
    assert len(result.calibration) == 4
    assert len(result.held_out_eval) == 6
    assert len(result.unused) == len(PAGE_IDS) - 10


def test_split_manifests_are_sorted_alphabetically() -> None:
    result = split_page_ids(PAGE_IDS, calibration_size=5, eval_size=None, seed=1337)
    assert list(result.calibration) == sorted(result.calibration)
    assert list(result.held_out_eval) == sorted(result.held_out_eval)


def test_split_rejects_oversized_calibration() -> None:
    with pytest.raises(ValueError, match="exceeds available"):
        split_page_ids(PAGE_IDS, calibration_size=999, eval_size=None, seed=1337)


def test_split_rejects_negative_sizes() -> None:
    with pytest.raises(ValueError, match="calibration_size"):
        split_page_ids(PAGE_IDS, calibration_size=-1, eval_size=None, seed=1337)
    with pytest.raises(ValueError, match="eval_size"):
        split_page_ids(PAGE_IDS, calibration_size=0, eval_size=-1, seed=1337)


def test_split_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        split_page_ids(["x", "x"], calibration_size=1, eval_size=None, seed=1337)
