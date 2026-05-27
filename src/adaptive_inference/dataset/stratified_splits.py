"""Stratified n=200 calibration/held-out splitter for scale-up v2.

Pure logic, no I/O. The CLI wrapper at
``scripts/scaleup/build_scaleup_splits.py`` reads ``PageRecord``s,
calls ``stratify_and_split``, and writes the JSON manifests.

Bucket rule (matches ``docs/specs/scaleup_v2_plan.md`` Stage 5):
* ``simple``       — ``row_count < 6``
* ``complex``      — ``6 <= row_count <= 15``
* ``very_complex`` — ``row_count >= 16`` OR ``has_merged_cells``

The split is *stratified at sample time and at partition time*: we draw
the n=200 subset proportionally from each bucket in the population, then
split that subset proportionally into 50 calibration / 150 held-out. If
a bucket has fewer than its proportional share, we take all of it and
top up the next-largest bucket.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Sequence

from .records import PageRecord


SCALEUP_V2_SEED_PHRASE = "SCALEUPv2"


def derive_seed(phrase: str = SCALEUP_V2_SEED_PHRASE) -> int:
    """Map a string seed phrase to a deterministic 64-bit int."""

    digest = hashlib.sha256(phrase.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def bucket_of(record: PageRecord) -> str:
    """Return the row-count bucket label for a record."""

    rows = record.row_count or 0
    if rows >= 16 or record.has_merged_cells:
        return "very_complex"
    if rows >= 6:
        return "complex"
    return "simple"


@dataclass(frozen=True)
class StratifiedSplit:
    calibration: tuple[str, ...]
    held_out: tuple[str, ...]
    bucket_stats: dict[str, "BucketStats"] = field(default_factory=dict)
    seed: int = 0
    seed_phrase: str = SCALEUP_V2_SEED_PHRASE


@dataclass(frozen=True)
class BucketStats:
    population: int
    sampled: int
    in_calibration: int
    in_held_out: int


def stratify_and_split(
    records: Sequence[PageRecord],
    *,
    n_total: int,
    n_calibration: int,
    seed: int | None = None,
    seed_phrase: str = SCALEUP_V2_SEED_PHRASE,
) -> StratifiedSplit:
    """Return a stratified calib/held-out split.

    Raises ``ValueError`` if the input has fewer than ``n_total``
    records or arguments are inconsistent.
    """

    if n_total <= 0 or n_calibration <= 0 or n_calibration >= n_total:
        raise ValueError(
            f"need 0 < n_calibration < n_total; got n_calibration={n_calibration}, "
            f"n_total={n_total}"
        )
    if len(records) < n_total:
        raise ValueError(
            f"only {len(records)} records available; need at least n_total={n_total}"
        )

    effective_seed = seed if seed is not None else derive_seed(seed_phrase)
    rng = random.Random(effective_seed)

    # Bucket the population.
    by_bucket: dict[str, list[PageRecord]] = {"simple": [], "complex": [], "very_complex": []}
    for r in records:
        by_bucket[bucket_of(r)].append(r)

    # Sort within bucket by page_id so the shuffle is reproducible regardless
    # of input order.
    for bucket in by_bucket.values():
        bucket.sort(key=lambda r: r.page_id)

    # Stratified sample n_total by proportional allocation (largest-remainder).
    population = sum(len(v) for v in by_bucket.values())
    quotas = _largest_remainder_allocation(
        targets={b: len(v) / population for b, v in by_bucket.items()},
        total=n_total,
        caps={b: len(v) for b, v in by_bucket.items()},
    )

    sampled: dict[str, list[PageRecord]] = {}
    for bucket, q in quotas.items():
        pool = list(by_bucket[bucket])
        rng.shuffle(pool)
        sampled[bucket] = pool[:q]

    # Stratified partition of the n_total sample into calibration / held_out.
    n_held_out = n_total - n_calibration
    calib_quotas = _largest_remainder_allocation(
        targets={b: len(v) / n_total for b, v in sampled.items()},
        total=n_calibration,
        caps={b: len(v) for b, v in sampled.items()},
    )

    calibration_ids: list[str] = []
    held_out_ids: list[str] = []
    stats: dict[str, BucketStats] = {}
    for bucket, pages in sampled.items():
        # Pages already shuffled. Take a deterministic slice.
        c_count = calib_quotas[bucket]
        calib_part = pages[:c_count]
        held_part = pages[c_count:]
        calibration_ids.extend(p.page_id for p in calib_part)
        held_out_ids.extend(p.page_id for p in held_part)
        stats[bucket] = BucketStats(
            population=len(by_bucket[bucket]),
            sampled=len(pages),
            in_calibration=len(calib_part),
            in_held_out=len(held_part),
        )

    # Final assertions: disjoint, correct sizes.
    calib_set = set(calibration_ids)
    held_set = set(held_out_ids)
    if calib_set & held_set:
        raise RuntimeError("internal error: calibration and held-out overlap")
    if len(calibration_ids) != n_calibration:
        raise RuntimeError(
            f"internal error: produced {len(calibration_ids)} calib, expected {n_calibration}"
        )
    if len(held_out_ids) != n_held_out:
        raise RuntimeError(
            f"internal error: produced {len(held_out_ids)} held-out, expected {n_held_out}"
        )

    return StratifiedSplit(
        calibration=tuple(sorted(calibration_ids)),
        held_out=tuple(sorted(held_out_ids)),
        bucket_stats=stats,
        seed=effective_seed,
        seed_phrase=seed_phrase,
    )


def _largest_remainder_allocation(
    *,
    targets: dict[str, float],
    total: int,
    caps: dict[str, int],
) -> dict[str, int]:
    """Proportional-quota allocation with per-bucket caps.

    Standard largest-remainder (Hare) method: floor each bucket's exact
    quota, distribute leftover seats by descending fractional remainder.
    Each bucket is capped at ``caps[bucket]``; overflow is spread to
    other buckets, repeating until the total is exact or no slack
    remains anywhere.
    """

    if abs(sum(targets.values()) - 1.0) > 1e-6:
        # Renormalize (a bucket may be empty).
        s = sum(targets.values())
        if s == 0:
            raise ValueError("all target proportions are zero")
        targets = {k: v / s for k, v in targets.items()}

    raw = {b: targets[b] * total for b in targets}
    alloc = {b: min(int(raw[b]), caps[b]) for b in targets}
    remainder = total - sum(alloc.values())

    # Distribute leftover by descending fractional part, skipping capped buckets.
    while remainder > 0:
        candidates = sorted(
            (b for b in targets if alloc[b] < caps[b]),
            key=lambda b: (raw[b] - int(raw[b])),
            reverse=True,
        )
        if not candidates:
            raise ValueError(
                f"cannot allocate {total} across buckets with caps={caps}; "
                f"capacity exhausted"
            )
        alloc[candidates[0]] += 1
        remainder -= 1
    return alloc
