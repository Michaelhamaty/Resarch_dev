"""Phase 4 smoke CLI: execute the adaptive inference pipeline end-to-end.

Loads an adaptive run config, feeds every page through the adaptive
orchestrator (low-budget parse -> verifier -> optional high-budget
reparse), and writes the first_pass/reparse/final artifact tree plus
one JSONL line per page to ``run.log.jsonl``. Mirrors the Phase 2 CLI
``scripts/main_runs/run_single_pass.py`` in style.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptive_inference.config.adaptive_runs import load_adaptive_run_config
from adaptive_inference.runner.adaptive import run_adaptive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to an adaptive run YAML (see configs/runs/smoke_adaptive.yaml).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted run: keep existing run.log.jsonl, skip "
            "pages already logged, append the rest. Without this flag, the "
            "log is truncated and every page is re-processed (Phase 4 default)."
        ),
    )
    args = parser.parse_args()

    cfg = load_adaptive_run_config(args.config)
    summary = run_adaptive(cfg, resume=args.resume)

    print(f"run_id          : {summary.run_id}")
    print(
        f"model           : {cfg.model_cfg.name} "
        f"(adapter={cfg.model_cfg.adapter_kind})"
    )
    print(
        f"budget_low      : {cfg.budget_low.name} "
        f"(max_tiles={cfg.budget_low.max_tiles})"
    )
    print(
        f"budget_high     : {cfg.budget_high.name} "
        f"(max_tiles={cfg.budget_high.max_tiles})"
    )
    print(f"prompt          : {cfg.prompt.id} v{cfg.prompt.version}")
    print(f"split           : {cfg.split_name}")
    print(f"pages_processed : {summary.pages_processed}")
    print(f"reparse_count   : {summary.reparse_count}")
    print(f"output_dir      : {summary.output_dir}")
    print(f"log_path        : {summary.log_path}")


if __name__ == "__main__":
    main()
