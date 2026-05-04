"""Score a Phase 2 run directory against an OmniDocBench ground-truth JSON.

Standalone CLI (no YAML config). Walks {run_dir}/pages/*.json sidecars,
scores each page via the cell-F1 + text-similarity scorer, and prints a
short summary. With --output-dir, also writes page_scores.json and a
markdown summary.

Low scores are honest signal, not failure — exit code is always 0 on a
successful run. Hard errors (missing run_dir, malformed GT) raise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptive_inference.analysis.run_scoring import score_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Phase 2 run directory (must contain pages/ and raw/).",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="JSON file mapping page_id to gold HTML.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="If set, write page_scores.json and summary.md here.",
    )
    args = parser.parse_args()

    result = score_run(
        run_dir=args.run_dir,
        ground_truth_path=args.ground_truth,
        output_dir=args.output_dir,
    )

    print(f"run_dir              : {result.run_dir}")
    print(f"ground_truth         : {result.ground_truth_path}")
    print(f"pages_total          : {result.pages_total}")
    print(f"pages_with_gold      : {result.pages_with_gold}")
    print(f"pages_with_parse_err : {result.pages_with_parse_error}")
    print(f"macro_cell_f1        : {result.macro_cell_f1:.4f}")
    print(f"macro_text_similarity: {result.macro_text_similarity:.4f}")
    if result.output_json_path is not None:
        print(f"output_json          : {result.output_json_path}")
        print(f"output_markdown      : {result.output_markdown_path}")
    else:
        print("output_dir           : (none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
