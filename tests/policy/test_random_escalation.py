"""Unit tests for the Phase 6 random-escalation policy."""

from __future__ import annotations

from adaptive_inference.policy.random_escalation import should_reparse_random


def test_zero_probability_never_reparses() -> None:
    for pid in ("page_0001", "page_0002", "page_0003"):
        assert (
            should_reparse_random(page_id=pid, seed=0, probability=0.0) is False
        )


def test_one_probability_always_reparses() -> None:
    for pid in ("page_0001", "page_0002", "page_0003"):
        assert (
            should_reparse_random(page_id=pid, seed=0, probability=1.0) is True
        )


def test_negative_probability_never_reparses() -> None:
    assert (
        should_reparse_random(page_id="page_0001", seed=0, probability=-0.5)
        is False
    )


def test_greater_than_one_probability_always_reparses() -> None:
    assert (
        should_reparse_random(page_id="page_0001", seed=0, probability=1.5)
        is True
    )


def test_decision_is_deterministic_for_same_seed_and_page() -> None:
    a = should_reparse_random(page_id="page_0005", seed=2, probability=0.5)
    b = should_reparse_random(page_id="page_0005", seed=2, probability=0.5)
    assert a == b


def test_decision_independent_of_iteration_order() -> None:
    # Reordering calls must not change per-page decisions, because each
    # draw is seeded from (seed, page_id) rather than from a shared stream.
    first_forward = should_reparse_random(
        page_id="page_0001", seed=7, probability=0.4
    )
    _ = should_reparse_random(page_id="page_0002", seed=7, probability=0.4)
    first_again = should_reparse_random(
        page_id="page_0001", seed=7, probability=0.4
    )
    assert first_forward == first_again


def test_different_seeds_give_different_population_decisions() -> None:
    page_ids = [f"page_{i:04d}" for i in range(200)]
    p = 0.5

    seed_a = [
        should_reparse_random(page_id=pid, seed=0, probability=p) for pid in page_ids
    ]
    seed_b = [
        should_reparse_random(page_id=pid, seed=1, probability=p) for pid in page_ids
    ]
    # Not identical across seeds (would require astronomical coincidence).
    assert seed_a != seed_b


def test_probability_half_approximately_half_reparses() -> None:
    page_ids = [f"page_{i:05d}" for i in range(2000)]
    hits = sum(
        should_reparse_random(page_id=pid, seed=0, probability=0.5)
        for pid in page_ids
    )
    # Lenient bounds — we only care the policy is not degenerate.
    assert 800 < hits < 1200
