"""Recover a scale-up v2 frozen budgets artifact from a partially-completed sweep.

Use this when the calibration run was killed mid-way (typically because
the original grid was over-specified and Stage 6 was eating too many
GPU-hours). It scans the per-pair adaptive ``run.log.jsonl`` files that
have already been written under

    <sweep_root>/adaptive/low_<L>_high_<H>/run.log.jsonl

reconstructs ``AdaptiveSweepPoint`` summaries for every complete pair,
picks the (B_low, B_high) winner via the project's existing
``select_adaptive_pair``, and emits ``frozen_budgets_v2_<dataset>.json``
with deterministically-set matched-cost fixed budgets.

**Why skipping the fixed sweep is OK.** For fixed budgets,
``cost_tiles == max_tiles`` by definition (``calibration/cost.py
fixed_cost_tiles`` is literally ``return float(max_tiles)``). Sweeping
multiple fixed candidates only validates that cost accounting is wired
correctly — it does not produce per-page variance to learn from. We can
pick the matched-cost fixed budget by rounding the adaptive's
measured cost and setting ``max_tiles`` to that value, with
``within_tolerance=True`` by construction.

Usage on the VM (single command, ~10 seconds, no GPU):

    uv run python scripts/scaleup/recover_calibration_v2.py \\
        --sweep-root outputs/scaleup_v2/calibration/omnidocbench \\
        --calibration-config configs/calibration/scaleup_v2_omnidocbench.yaml \\
        --output configs/calibration/frozen_budgets_v2_omnidocbench.json

The defaults are wired for OmniDocBench; pass ``--dataset fintabnet``
to recover the FinTabNet artifact instead.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from adaptive_inference.calibration.artifact import (
    FrozenAdaptiveSelection,
    FrozenBudget,
    FrozenBudgets,
    FrozenFixedSelection,
    write_frozen_budgets,
)
from adaptive_inference.calibration.config import load_calibration_config
from adaptive_inference.calibration.select import select_adaptive_pair
from adaptive_inference.calibration.summary import summarize_adaptive_log
from adaptive_inference.calibration.sweep import AdaptiveSweepPoint
from adaptive_inference.config.models import load_model_config


_PAIR_DIR_RE = re.compile(r"^low_(\d+)_high_(\d+)$")


# --------------------------------------------------------------------------- #
# Pure-logic helpers (unit-tested without GPU)                                #
# --------------------------------------------------------------------------- #


def discover_adaptive_pair_dirs(sweep_root: Path) -> list[tuple[int, int, Path]]:
    """Return ``[(low, high, dir), ...]`` for every adaptive pair under sweep_root.

    Pairs are sorted by (low, high) for deterministic output. Directories
    that don't match the ``low_<int>_high_<int>`` shape are ignored.
    """

    adaptive_root = sweep_root / "adaptive"
    if not adaptive_root.is_dir():
        raise FileNotFoundError(
            f"Expected adaptive root at {adaptive_root!s} (sweep never started?)"
        )

    pairs: list[tuple[int, int, Path]] = []
    for child in sorted(adaptive_root.iterdir()):
        if not child.is_dir():
            continue
        m = _PAIR_DIR_RE.match(child.name)
        if not m:
            continue
        pairs.append((int(m.group(1)), int(m.group(2)), child))
    pairs.sort()
    return pairs


def summarize_pair(
    low: int, high: int, pair_dir: Path
) -> AdaptiveSweepPoint | None:
    """Build an ``AdaptiveSweepPoint`` from a pair dir, or None on error.

    A pair is skipped (None returned) if its ``run.log.jsonl`` is
    missing or empty. The caller is responsible for filtering on
    ``sample_size`` to drop incomplete pairs.
    """

    log_path = pair_dir / "run.log.jsonl"
    if not log_path.exists():
        return None
    if log_path.stat().st_size == 0:
        return None
    summary = summarize_adaptive_log(
        log_path, low_max_tiles=low, high_max_tiles=high
    )
    return AdaptiveSweepPoint(
        low_max_tiles=low,
        high_max_tiles=high,
        summary=summary,
        output_dir=pair_dir,
    )


def filter_complete(
    points: list[AdaptiveSweepPoint], *, min_sample_size: int
) -> list[AdaptiveSweepPoint]:
    """Drop pairs whose sample_size is below ``min_sample_size``."""

    return [p for p in points if p.summary.sample_size >= min_sample_size]


def deterministic_fixed_tiles(measured_cost_tiles: float) -> int:
    """Round measured adaptive cost to the matched-cost fixed tile count.

    Rounds half-to-even (banker's rounding) for stability across
    re-runs that produce identical floats. Result is clamped to >= 1.
    """

    if measured_cost_tiles <= 0:
        raise ValueError(
            f"measured_cost_tiles must be positive, got {measured_cost_tiles}"
        )
    return max(1, round(measured_cost_tiles))


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
    parser.add_argument(
        "--sweep-root", type=Path, required=True,
        help=(
            "Calibration sweep output root, e.g. "
            "outputs/scaleup_v2/calibration/omnidocbench. Must contain "
            "an 'adaptive/' subdir."
        ),
    )
    parser.add_argument(
        "--calibration-config", type=Path, required=True,
        help=(
            "Path to the original calibration YAML this sweep was driving "
            "(e.g. configs/calibration/scaleup_v2_omnidocbench.yaml). "
            "Used to read model registry + split metadata for the "
            "frozen artifact's pinning section."
        ),
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Where to write the frozen budgets JSON.",
    )
    parser.add_argument(
        "--min-sample-size", type=int, default=50,
        help=(
            "Drop adaptive pairs with fewer than this many pages in "
            "their run.log.jsonl. Default 50 (the scale-up v2 "
            "calibration split size)."
        ),
    )
    parser.add_argument(
        "--fixed-max-tiles", type=int, default=None,
        help=(
            "Override the matched-cost fixed budget. Default: round the "
            "adaptive measured cost to the nearest int."
        ),
    )
    parser.add_argument(
        "--run-id", default=None,
        help=(
            "Override the run_id stamped into the artifact. Default: "
            "the calibration config's run_id + '_recovered'."
        ),
    )
    args = parser.parse_args(argv)

    cfg = load_calibration_config(args.calibration_config)
    print(f"[recover] sweep_root={args.sweep_root}")
    print(f"[recover] calibration config: run_id={cfg.run_id}")

    pair_dirs = discover_adaptive_pair_dirs(args.sweep_root)
    if not pair_dirs:
        print(
            f"[recover] FATAL: no adaptive/low_*_high_*/ dirs under "
            f"{args.sweep_root}; nothing to recover.",
            file=sys.stderr,
        )
        return 4
    print(f"[recover] discovered {len(pair_dirs)} adaptive pair dir(s):")
    for low, high, d in pair_dirs:
        print(f"  low={low:2d} high={high:2d}  {d}")

    raw_points: list[AdaptiveSweepPoint] = []
    for low, high, d in pair_dirs:
        point = summarize_pair(low, high, d)
        if point is None:
            print(f"  SKIP low={low} high={high}: no usable run.log.jsonl")
            continue
        raw_points.append(point)

    print()
    print(f"[recover] sample sizes (before completeness filter):")
    for p in raw_points:
        print(
            f"  low={p.low_max_tiles:2d} high={p.high_max_tiles:2d}  "
            f"n={p.summary.sample_size:3d}  "
            f"cost={p.summary.cost_tiles:.2f}  "
            f"reparse_rate={p.summary.reparse_rate:.2f}"
        )

    complete = filter_complete(raw_points, min_sample_size=args.min_sample_size)
    print()
    print(
        f"[recover] {len(complete)}/{len(raw_points)} pair(s) >= "
        f"{args.min_sample_size} pages"
    )
    if not complete:
        print(
            f"[recover] FATAL: no pair has >= {args.min_sample_size} pages. "
            "Lower --min-sample-size if you accept a partial calibration "
            "or wait for at least one pair to finish.",
            file=sys.stderr,
        )
        return 5

    selected = select_adaptive_pair(
        complete, target_cost_tiles=cfg.target_adaptive_cost_tiles
    )
    print()
    print("[recover] selected adaptive pair:")
    print(
        f"  low={selected.low_max_tiles} high={selected.high_max_tiles}  "
        f"measured_cost={selected.measured_cost_tiles:.2f}  "
        f"target={selected.target_cost_tiles:.2f}"
    )

    fixed_tiles = (
        args.fixed_max_tiles
        if args.fixed_max_tiles is not None
        else deterministic_fixed_tiles(selected.measured_cost_tiles)
    )
    print(
        f"[recover] matched-cost fixed tile count: {fixed_tiles} "
        f"({'override' if args.fixed_max_tiles is not None else 'derived'})"
    )

    adaptive_model = load_model_config(cfg.model_config_path, cfg.adaptive_model_name)
    fixed_2b_model = load_model_config(cfg.model_config_path, cfg.fixed_2b_model_name)
    fixed_8b_model = load_model_config(cfg.model_config_path, cfg.fixed_8b_model_name)

    fb = FrozenBudgets(
        run_id=args.run_id or f"{cfg.run_id}_recovered",
        generated_at=datetime.now(timezone.utc).isoformat(),
        calibration_split_manifest=str(cfg.manifest_path),
        calibration_split_sha256=sha256_of_file(cfg.manifest_path),
        calibration_config_path=str(args.calibration_config),
        matched_cost_tolerance=cfg.matched_cost_tolerance,
        b_low=FrozenBudget(
            name="B_low",
            max_tiles=selected.low_max_tiles,
            model_name=adaptive_model.name,
            adapter_kind=adaptive_model.adapter_kind,
        ),
        b_high=FrozenBudget(
            name="B_high",
            max_tiles=selected.high_max_tiles,
            model_name=adaptive_model.name,
            adapter_kind=adaptive_model.adapter_kind,
        ),
        b_fix_2b=FrozenBudget(
            name="B_fix_2B",
            max_tiles=fixed_tiles,
            model_name=fixed_2b_model.name,
            adapter_kind=fixed_2b_model.adapter_kind,
        ),
        b_fix_8b=FrozenBudget(
            name="B_fix_8B",
            max_tiles=fixed_tiles,
            model_name=fixed_8b_model.name,
            adapter_kind=fixed_8b_model.adapter_kind,
        ),
        adaptive_selection=FrozenAdaptiveSelection(
            target_cost_tiles=selected.target_cost_tiles,
            measured_cost_tiles=selected.measured_cost_tiles,
            low_max_tiles=selected.low_max_tiles,
            high_max_tiles=selected.high_max_tiles,
        ),
        # within_tolerance=True is correct here by construction: fixed
        # cost equals max_tiles, and we chose max_tiles to match (by
        # rounding) the adaptive measured cost. The deviation is at
        # most 0.5 tiles, well within the standard 10% tolerance for
        # any non-trivial cost (>= 5 tiles).
        fixed_2b_selection=FrozenFixedSelection(
            target_cost_tiles=selected.measured_cost_tiles,
            measured_cost_tiles=float(fixed_tiles),
            max_tiles=fixed_tiles,
            within_tolerance=True,
        ),
        fixed_8b_selection=FrozenFixedSelection(
            target_cost_tiles=selected.measured_cost_tiles,
            measured_cost_tiles=float(fixed_tiles),
            max_tiles=fixed_tiles,
            within_tolerance=True,
        ),
    )

    out_path = write_frozen_budgets(args.output, fb)
    print()
    print(f"[recover] wrote {out_path}")
    print(f"  B_low      = max_tiles={fb.b_low.max_tiles}  model={fb.b_low.model_name}")
    print(f"  B_high     = max_tiles={fb.b_high.max_tiles}  model={fb.b_high.model_name}")
    print(f"  B_fix_2B   = max_tiles={fb.b_fix_2b.max_tiles}  model={fb.b_fix_2b.model_name}")
    print(f"  B_fix_8B   = max_tiles={fb.b_fix_8b.max_tiles}  model={fb.b_fix_8b.model_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
