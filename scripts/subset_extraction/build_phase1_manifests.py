"""CLI entrypoint: run Phase 1 freeze and print a summary.

Usage:
    uv run python scripts/subset_extraction/build_phase1_manifests.py \
        --config configs/dataset/phase1.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptive_inference.dataset import freeze_phase1_universe


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the Phase 1 experiment universe.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the Phase 1 YAML config (e.g. configs/dataset/phase1.yaml).",
    )
    args = parser.parse_args()

    result = freeze_phase1_universe(args.config)

    print(f"Phase 1 freeze complete ({args.config})")
    print(f"  dataset           : {result.pinning.dataset_name}")
    print(f"  snapshot_version  : {result.pinning.snapshot_version}")
    print(f"  eval_code_commit  : {result.pinning.eval_code_commit}")
    print(f"  manifests_dir     : {result.manifests_dir}")
    print(f"  eval_universe     : {len(result.eval_universe_ids)} pages")
    print(f"  hard_subset       : {len(result.hard_subset_ids)} pages")
    print(f"  calibration_split : {len(result.split.calibration)} pages")
    print(f"  held_out_eval     : {len(result.split.held_out_eval)} pages")
    print(f"  unused            : {len(result.split.unused)} pages")
    for kind, path in result.written_paths.items():
        print(f"  wrote {kind:<22} -> {path}")


if __name__ == "__main__":
    main()
