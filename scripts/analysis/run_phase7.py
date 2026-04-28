"""Phase 7 CLI: ingest Phase 6 outputs and produce analysis artifacts.

Reads the Phase 7 YAML, runs the analysis pipeline, prints a short
summary, and exits non-zero if any audit check failed. Audit warns
(e.g. the explicit stub-adapter gate) do NOT cause a non-zero exit;
they are surfaced in the printed summary instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptive_inference.analysis.config import load_phase7_config
from adaptive_inference.analysis.runner import run_phase7


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a Phase 7 YAML (see configs/analysis/phase7.yaml).",
    )
    args = parser.parse_args()

    cfg = load_phase7_config(args.config)
    result = run_phase7(cfg)

    print(f"output_dir   : {result.output_dir}")
    print(f"audit checks : {len(result.audit.checks)}")
    for c in result.audit.checks:
        print(f"  - {c.name:<48} {c.status:>4}  {c.detail}")
    print(f"any_failed   : {result.audit.any_failed}")
    print(f"any_warned   : {result.audit.any_warned}")
    return 1 if result.audit.any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
