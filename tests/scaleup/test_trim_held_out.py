"""Pure-logic tests for scripts/scaleup/trim_held_out.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_trim_module():
    repo_root = Path(__file__).resolve().parents[2]
    spec_path = repo_root / "scripts" / "scaleup" / "trim_held_out.py"
    spec = importlib.util.spec_from_file_location("trim_held_out", spec_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["trim_held_out"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def trim():
    return _load_trim_module()


def test_proportional_targets_sums_to_target_n(trim):
    bucket_to_ids = {
        "simple": [f"s{i}" for i in range(19)],
        "complex": [f"c{i}" for i in range(40)],
        "very_complex": [f"v{i}" for i in range(91)],
    }
    targets = trim.proportional_targets(bucket_to_ids, target_n=100)
    assert sum(targets.values()) == 100


def test_proportional_targets_approx_proportional(trim):
    bucket_to_ids = {
        "simple": [f"s{i}" for i in range(19)],
        "complex": [f"c{i}" for i in range(40)],
        "very_complex": [f"v{i}" for i in range(91)],
    }
    targets = trim.proportional_targets(bucket_to_ids, target_n=100)
    # 19/150 of 100 = 12.67 -> 12 or 13
    assert targets["simple"] in (12, 13)
    # 40/150 of 100 = 26.67 -> 26 or 27
    assert targets["complex"] in (26, 27)
    # 91/150 of 100 = 60.67 -> 60 or 61
    assert targets["very_complex"] in (60, 61)


def test_proportional_targets_caps_at_population(trim):
    bucket_to_ids = {
        "tiny": ["a", "b"],
        "huge": [f"h{i}" for i in range(100)],
    }
    targets = trim.proportional_targets(bucket_to_ids, target_n=50)
    assert targets["tiny"] <= 2  # cannot exceed population
    assert sum(targets.values()) == 50


def test_proportional_targets_rejects_invalid_inputs(trim):
    with pytest.raises(ValueError):
        trim.proportional_targets({"a": ["x"]}, target_n=0)
    with pytest.raises(ValueError):
        trim.proportional_targets({}, target_n=10)
    with pytest.raises(ValueError):
        trim.proportional_targets({"a": ["x"]}, target_n=10)  # exceeds total


def test_stratified_sample_preserves_proportions(trim):
    bucket_to_ids = {
        "a": [f"a{i}" for i in range(30)],
        "b": [f"b{i}" for i in range(60)],
        "c": [f"c{i}" for i in range(60)],
    }
    sampled = trim.stratified_sample(bucket_to_ids, target_n=50, seed=1234)
    assert sum(len(v) for v in sampled.values()) == 50
    # 30/150 of 50 = 10
    assert len(sampled["a"]) == 10
    # 60/150 of 50 = 20
    assert len(sampled["b"]) == 20
    assert len(sampled["c"]) == 20


def test_stratified_sample_is_deterministic_for_same_seed(trim):
    bucket_to_ids = {"a": [f"a{i}" for i in range(50)]}
    s1 = trim.stratified_sample(bucket_to_ids, target_n=10, seed=42)
    s2 = trim.stratified_sample(bucket_to_ids, target_n=10, seed=42)
    assert s1 == s2


def test_stratified_sample_differs_for_different_seeds(trim):
    bucket_to_ids = {"a": [f"a{i}" for i in range(50)]}
    s1 = trim.stratified_sample(bucket_to_ids, target_n=10, seed=42)
    s2 = trim.stratified_sample(bucket_to_ids, target_n=10, seed=99)
    assert s1 != s2


def test_stratified_sample_output_sorted(trim):
    bucket_to_ids = {"a": [f"a{i}" for i in reversed(range(20))]}
    sampled = trim.stratified_sample(bucket_to_ids, target_n=5, seed=1)
    assert sampled["a"] == sorted(sampled["a"])


def test_sha256_of_file(trim, tmp_path):
    p = tmp_path / "x"
    p.write_text("hello\n", encoding="utf-8")
    assert (
        trim.sha256_of_file(p)
        == "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
    )
