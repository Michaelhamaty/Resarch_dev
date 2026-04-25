"""Phase 6 CLI: execute every required system on the held-out eval split.

Reads the Phase 6 YAML, loads the Phase 5 frozen artifact (read-only),
runs each system (adaptive 2B, fixed 2B low, fixed 2B matched, random 2B
seeds 0/1/2, fixed 8B matched) under the same harness, and writes a
run manifest.

Exits non-zero if any system failed so CI and batch scripts can fail
loudly. Skipped-stub-8B does NOT count as a failure — it is an explicit
policy decision recorded in the manifest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptive_inference.experiment.config import load_phase6_config
from adaptive_inference.experiment.runner import run_phase6


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a Phase 6 YAML (see configs/experiment/phase6.yaml).",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=None,
        help=(
            "Optional filter: one or more family ids "
            "(adaptive_2b, fixed_2b_low, fixed_2b_matched, random_2b, "
            "fixed_8b_matched) or specific system ids like random_2b_seed1."
        ),
    )
    parser.add_argument(
        "--allow-stubbed-8b",
        action="store_true",
        help=(
            "Run fixed_8b_matched even when B_fix_8B.adapter_kind == 'stub'. "
            "Without this flag the 8B system is skipped to avoid silently "
            "faking the headline upper-bound baseline."
        ),
    )
    args = parser.parse_args()

    cfg = load_phase6_config(args.config)
    filters = tuple(args.systems) if args.systems else None
    result = run_phase6(
        cfg,
        allow_stubbed_8b=args.allow_stubbed_8b,
        systems_filter=filters,
    )

    print(f"run_set_id    : {cfg.run_set_id}")
    print(f"manifest_path : {result.manifest_path}")
    print(f"systems_run   : {len(result.entries)}")
    for entry in result.entries:
        line = f"  - {entry.system_id:<24} status={entry.status}"
        if entry.pages_processed is not None:
            line += f" pages={entry.pages_processed}"
        if entry.reparse_count is not None:
            line += f" reparses={entry.reparse_count}"
        if entry.status == "failed" and entry.error:
            first_line = entry.error.splitlines()[0] if entry.error else ""
            line += f" error={first_line}"
        if entry.status == "skipped_stub_8b":
            line += " (skipped: stub 8B without --allow-stubbed-8b)"
        print(line)

    return 1 if result.any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
