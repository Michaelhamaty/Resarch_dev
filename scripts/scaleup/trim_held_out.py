"""Trim a scale-up v2 held-out split to a smaller stratified subset.

Used to invoke the plan's Risk #4 fallback ("cut held-out to 100/dataset
if the main sweep doesn't finish overnight"). Reads an existing
held-out manifest (e.g. ``data/splits/scaleup_v2/fintabnet/held_out.json``)
plus the source records, stratified-samples ``--target-n`` pages
preserving the original bucket proportions, and writes a parallel
manifest with a ``trimmed_from`` field for audit traceability.

Determinism: seeded by ``--seed-phrase`` (default ``SCALEUPv2_trim``).
Re-running the same command against unchanged inputs produces a
byte-identical output file.

Usage::

    uv run python scripts/scaleup/trim_held_out.py \\
        --source data/splits/scaleup_v2/fintabnet/held_out.json \\
        --records data/fintabnet/records.json \\
        --target-n 100 \\
        --output data/splits/scaleup_v2/fintabnet/held_out_100.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from adaptive_inference.dataset.records import load_page_records
from adaptive_inference.dataset.stratified_splits import (
    SCALEUP_V2_SEED_PHRASE,
    bucket_of,
    derive_seed,
)


SCHEMA_VERSION = 1
TRIM_SEED_PHRASE = f"{SCALEUP_V2_SEED_PHRASE}_trim"


# --------------------------------------------------------------------------- #
# Pure-logic helpers (unit-tested without I/O)                                #
# --------------------------------------------------------------------------- #


def proportional_targets(
    bucket_to_ids: dict[str, list[str]], target_n: int
) -> dict[str, int]:
    """Return per-bucket target counts proportional to bucket population.

    Uses largest-remainder allocation so the sum of allocations equals
    exactly ``target_n``. If a bucket has fewer page_ids than its
    proportional target, it caps at its population and the surplus is
    redistributed via the remainder ordering.
    """

    if target_n <= 0:
        raise ValueError(f"target_n must be positive, got {target_n}")

    total = sum(len(ids) for ids in bucket_to_ids.values())
    if total == 0:
        raise ValueError("bucket_to_ids is empty; nothing to allocate")
    if target_n > total:
        raise ValueError(
            f"target_n={target_n} exceeds total available page_ids ({total})"
        )

    exact = {b: target_n * len(ids) / total for b, ids in bucket_to_ids.items()}
    floors = {b: int(v) for b, v in exact.items()}
    remainders = {b: exact[b] - floors[b] for b in bucket_to_ids}

    allocated = sum(floors.values())
    surplus = target_n - allocated

    # Distribute the surplus to the buckets with the largest fractional
    # remainder. Tie-break alphabetically by bucket name for stability.
    order = sorted(
        bucket_to_ids.keys(), key=lambda b: (-remainders[b], b)
    )
    for b in order[:surplus]:
        floors[b] += 1

    # Cap at population; pour overflow into other buckets in the same order.
    targets = dict(floors)
    for b, ids in bucket_to_ids.items():
        if targets[b] > len(ids):
            overflow = targets[b] - len(ids)
            targets[b] = len(ids)
            for other in order:
                if other == b:
                    continue
                room = len(bucket_to_ids[other]) - targets[other]
                if room <= 0:
                    continue
                take = min(room, overflow)
                targets[other] += take
                overflow -= take
                if overflow == 0:
                    break

    return targets


def stratified_sample(
    bucket_to_ids: dict[str, list[str]],
    target_n: int,
    *,
    seed: int,
) -> dict[str, list[str]]:
    """Return ``{bucket: sampled_ids}`` preserving proportions.

    Sampling within each bucket is uniform-without-replacement, seeded
    deterministically by ``seed`` (mixed with the bucket name so two
    buckets don't draw identical indices into different populations).
    """

    targets = proportional_targets(bucket_to_ids, target_n)
    out: dict[str, list[str]] = {}
    for bucket in sorted(bucket_to_ids):
        ids = sorted(bucket_to_ids[bucket])
        rng = random.Random(_mix_seed(seed, bucket))
        chosen = rng.sample(ids, targets[bucket])
        out[bucket] = sorted(chosen)
    return out


def _mix_seed(base: int, bucket: str) -> int:
    payload = f"{base}::{bucket}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Existing held-out manifest to trim from.")
    parser.add_argument("--records", type=Path, required=True,
                        help="records.json for the same dataset.")
    parser.add_argument("--target-n", type=int, required=True,
                        help="How many page_ids to keep in the trimmed split.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Where to write the trimmed manifest.")
    parser.add_argument("--seed-phrase", default=TRIM_SEED_PHRASE,
                        help=f"Seed phrase (default: {TRIM_SEED_PHRASE}).")
    args = parser.parse_args(argv)

    source_payload = json.loads(args.source.read_text(encoding="utf-8"))
    source_ids: list[str] = list(source_payload.get("page_ids") or [])
    if not source_ids:
        print(f"FATAL: no page_ids in {args.source}", file=__import__("sys").stderr)
        return 2
    if args.target_n >= len(source_ids):
        print(
            f"FATAL: target_n={args.target_n} >= source size ({len(source_ids)}); "
            "nothing to trim.",
            file=__import__("sys").stderr,
        )
        return 2

    records = load_page_records(args.records)
    by_id = {r.page_id: r for r in records}
    missing = [pid for pid in source_ids if pid not in by_id]
    if missing:
        print(
            f"FATAL: {len(missing)} page_ids in {args.source} not found in "
            f"{args.records} (first 3: {missing[:3]})",
            file=__import__("sys").stderr,
        )
        return 3

    bucket_to_ids: dict[str, list[str]] = defaultdict(list)
    for pid in source_ids:
        bucket_to_ids[bucket_of(by_id[pid])].append(pid)
    bucket_to_ids = dict(bucket_to_ids)

    seed = derive_seed(args.seed_phrase)
    sampled = stratified_sample(bucket_to_ids, args.target_n, seed=seed)
    page_ids = sorted(p for chosen in sampled.values() for p in chosen)

    bucket_stats = {
        bucket: {
            "source_in_held_out": len(bucket_to_ids[bucket]),
            "kept": len(sampled[bucket]),
        }
        for bucket in sorted(bucket_to_ids)
    }

    header = {
        "schema_version": SCHEMA_VERSION,
        "dataset": source_payload.get("dataset"),
        "split": "held_out_trimmed",
        "n_target": args.target_n,
        "n_kept": len(page_ids),
        "seed_phrase": args.seed_phrase,
        "seed": seed,
        "trimmed_from": str(args.source),
        "trimmed_from_sha256": sha256_of_file(args.source),
        "records_path": str(args.records),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bucket_stats": bucket_stats,
        "page_ids": page_ids,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(header, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"wrote {args.output} (n={len(page_ids)})")
    for bucket, stats in bucket_stats.items():
        print(
            f"  bucket={bucket:14s} source={stats['source_in_held_out']:3d} "
            f"kept={stats['kept']:3d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
