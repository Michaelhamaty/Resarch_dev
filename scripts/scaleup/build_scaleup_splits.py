"""Scale-up v2 Stage 5: build stratified 50/150 splits for each dataset.

For each dataset (omnidocbench, fintabnet), reads its ``records.json``
and writes two manifests:

* ``data/splits/scaleup_v2/<dataset>/calibration.json``  — 50 page IDs
* ``data/splits/scaleup_v2/<dataset>/held_out.json``     — 150 page IDs

Bucketing is by row-count (simple / complex / very_complex). The split
is deterministic given the seed phrase ``SCALEUPv2`` and the records
file's contents.

Run on the VM after both fixtures are built::

    python scripts/scaleup/build_scaleup_splits.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from adaptive_inference.dataset.records import load_page_records
from adaptive_inference.dataset.stratified_splits import (
    SCALEUP_V2_SEED_PHRASE,
    stratify_and_split,
)


SCHEMA_VERSION = 1

DEFAULT_DATASETS = {
    "omnidocbench": Path("data/omnidocbench/records.json"),
    "fintabnet": Path("data/fintabnet/records.json"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", action="append", default=None,
        help="Only build splits for the named dataset(s); repeatable.",
    )
    parser.add_argument(
        "--records-path", action="append", default=None,
        help="Override a dataset's records.json path (paired with --dataset).",
    )
    parser.add_argument("--n-total", type=int, default=200)
    parser.add_argument("--n-calibration", type=int, default=50)
    parser.add_argument(
        "--out-root", type=Path, default=Path("data/splits/scaleup_v2"),
    )
    parser.add_argument(
        "--seed-phrase", default=SCALEUP_V2_SEED_PHRASE,
        help="Override the seed phrase (default: SCALEUPv2).",
    )
    args = parser.parse_args()

    if args.dataset and args.records_path:
        if len(args.dataset) != len(args.records_path):
            print(
                "--dataset and --records-path must be paired (same count).",
            )
            return 2
        datasets = dict(zip(args.dataset, [Path(p) for p in args.records_path]))
    elif args.dataset:
        datasets = {name: DEFAULT_DATASETS[name] for name in args.dataset}
    else:
        datasets = DEFAULT_DATASETS

    git_sha = _git_sha()
    rc = 0
    for name, records_path in datasets.items():
        if not records_path.exists():
            print(f"[{name}] SKIP — records file not found: {records_path}")
            rc = 1
            continue
        rc = max(rc, _build_one_dataset(
            name=name,
            records_path=records_path,
            n_total=args.n_total,
            n_calibration=args.n_calibration,
            out_root=args.out_root,
            seed_phrase=args.seed_phrase,
            git_sha=git_sha,
        ))
    return rc


def _build_one_dataset(
    *,
    name: str,
    records_path: Path,
    n_total: int,
    n_calibration: int,
    out_root: Path,
    seed_phrase: str,
    git_sha: str,
) -> int:
    records = load_page_records(records_path)
    if len(records) < n_total:
        print(
            f"[{name}] insufficient records: have {len(records)}, need {n_total}. "
            "Rebuild the fixture with a larger --limit.",
        )
        return 3
    print(f"[{name}] loaded {len(records)} records from {records_path}")

    result = stratify_and_split(
        records,
        n_total=n_total,
        n_calibration=n_calibration,
        seed_phrase=seed_phrase,
    )

    records_sha = _sha256_of_file(records_path)
    header = {
        "schema_version": SCHEMA_VERSION,
        "dataset": name,
        "n_total": n_total,
        "n_calibration": n_calibration,
        "n_held_out": n_total - n_calibration,
        "seed_phrase": result.seed_phrase,
        "seed": result.seed,
        "sampling_rule": (
            "row-count buckets simple<6, complex 6-15, very_complex>=16 or merged; "
            "largest-remainder stratified sample then stratified disjoint split."
        ),
        "records_path": str(records_path),
        "records_sha256": records_sha,
        "git_sha": git_sha,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bucket_stats": {
            bucket: {
                "population": s.population,
                "sampled": s.sampled,
                "in_calibration": s.in_calibration,
                "in_held_out": s.in_held_out,
            }
            for bucket, s in result.bucket_stats.items()
        },
    }

    dataset_dir = out_root / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(
        dataset_dir / "calibration.json",
        header=header | {"split": "calibration"},
        page_ids=result.calibration,
    )
    _write_manifest(
        dataset_dir / "held_out.json",
        header=header | {"split": "held_out"},
        page_ids=result.held_out,
    )
    print(f"[{name}] wrote {dataset_dir}/calibration.json (n={len(result.calibration)})")
    print(f"[{name}] wrote {dataset_dir}/held_out.json    (n={len(result.held_out)})")
    for bucket, s in result.bucket_stats.items():
        print(
            f"  bucket={bucket:14s} pop={s.population:4d} sampled={s.sampled:3d} "
            f"calib={s.in_calibration:3d} held_out={s.in_held_out:3d}"
        )
    return 0


def _write_manifest(path: Path, *, header: dict, page_ids: tuple[str, ...]) -> None:
    payload = {**header, "page_ids": list(page_ids)}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
