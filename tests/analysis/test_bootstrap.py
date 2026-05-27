"""Tests for page-level bootstrap CI."""

from __future__ import annotations

import pytest

from adaptive_inference.analysis.bootstrap import bootstrap_ci


def test_constant_array_zero_width():
    res = bootstrap_ci([0.7] * 20, n_resamples=500, seed=1)
    assert res.mean == pytest.approx(0.7)
    assert res.lo == pytest.approx(0.7)
    assert res.hi == pytest.approx(0.7)


def test_mean_is_arithmetic_mean():
    res = bootstrap_ci([0.0, 1.0, 0.5], n_resamples=100, seed=1)
    assert res.mean == pytest.approx(0.5)


def test_ci_contains_mean_for_random_input():
    import random
    rng = random.Random(0)
    data = [rng.gauss(0.6, 0.1) for _ in range(40)]
    res = bootstrap_ci(data, n_resamples=2000, seed=42)
    assert res.lo <= res.mean <= res.hi
    # 95% CI on a sample of 40 with sigma=0.1 should be very tight.
    assert res.hi - res.lo < 0.15


def test_seed_determinism():
    data = [0.1, 0.9, 0.5, 0.7, 0.3]
    a = bootstrap_ci(data, n_resamples=500, seed=7)
    b = bootstrap_ci(data, n_resamples=500, seed=7)
    assert a == b


def test_singleton_returns_point_value():
    res = bootstrap_ci([0.42], n_resamples=100, seed=1)
    assert res.lo == res.hi == res.mean == 0.42
    assert res.n == 1


def test_empty_raises():
    with pytest.raises(ValueError):
        bootstrap_ci([], n_resamples=10)


def test_bad_ci_raises():
    with pytest.raises(ValueError):
        bootstrap_ci([0.5], ci=1.5)
    with pytest.raises(ValueError):
        bootstrap_ci([0.5], ci=0.0)


def test_bad_n_resamples_raises():
    with pytest.raises(ValueError):
        bootstrap_ci([0.5], n_resamples=0)
