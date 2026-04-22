"""Deterministic split of page IDs into calibration and held-out eval.

Determinism recipe: sort alphabetically → seed ``random.Random`` → shuffle →
slice. Same inputs + same seed always yield the same split, independent of
dict or filesystem ordering.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SplitResult:
    calibration: tuple[str, ...]
    held_out_eval: tuple[str, ...]
    unused: tuple[str, ...]
    seed: int


def split_page_ids(
    page_ids: list[str],
    *,
    calibration_size: int,
    eval_size: int | None,
    seed: int,
) -> SplitResult:
    if calibration_size < 0:
        raise ValueError(f"calibration_size must be >= 0, got {calibration_size}")
    if eval_size is not None and eval_size < 0:
        raise ValueError(f"eval_size must be >= 0, got {eval_size}")

    sorted_ids = sorted(set(page_ids))
    if len(sorted_ids) != len(page_ids):
        raise ValueError("split_page_ids received duplicate page IDs")

    resolved_eval_size = (
        len(sorted_ids) - calibration_size if eval_size is None else eval_size
    )
    if resolved_eval_size < 0:
        raise ValueError(
            "calibration_size exceeds available page IDs "
            f"(have {len(sorted_ids)}, calibration_size={calibration_size})"
        )
    if calibration_size + resolved_eval_size > len(sorted_ids):
        raise ValueError(
            "calibration_size + eval_size exceeds available page IDs "
            f"({calibration_size} + {resolved_eval_size} > {len(sorted_ids)})"
        )

    rng = random.Random(seed)
    shuffled = list(sorted_ids)
    rng.shuffle(shuffled)

    cal = shuffled[:calibration_size]
    held_out = shuffled[calibration_size : calibration_size + resolved_eval_size]
    unused = shuffled[calibration_size + resolved_eval_size :]

    return SplitResult(
        calibration=tuple(sorted(cal)),
        held_out_eval=tuple(sorted(held_out)),
        unused=tuple(sorted(unused)),
        seed=seed,
    )
