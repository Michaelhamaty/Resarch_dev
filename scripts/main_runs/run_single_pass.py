"""Phase 2 smoke CLI: execute the single-pass inference pipeline.

Loads a run config, feeds every page listed in its manifest through the
configured adapter, and writes per-page artifacts plus a JSONL runtime
log. Mirrors the style of the Phase 1 CLI
(``scripts/subset_extraction/build_phase1_manifests.py``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptive_inference.config.runs import load_run_config
from adaptive_inference.runner.single_pass import run_single_pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a run YAML (see configs/runs/).",
    )
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    summary = run_single_pass(cfg)

    print(f"run_id          : {summary.run_id}")
    print(f"model           : {cfg.model_cfg.name} (adapter={cfg.model_cfg.adapter_kind})")
    print(f"budget          : {cfg.budget.name} (max_tiles={cfg.budget.max_tiles})")
    print(f"prompt          : {cfg.prompt.id} v{cfg.prompt.version}")
    print(f"split           : {cfg.split_name}")
    print(f"pages_processed : {summary.pages_processed}")
    print(f"output_dir      : {summary.output_dir}")
    print(f"log_path        : {summary.log_path}")


if __name__ == "__main__":
    main()
