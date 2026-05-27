"""Scale-up v2 Stage 4 G4 gate: 5-page FinTabNet smoke through real 2B.

Run on the VM after ``build_fintabnet_fixture.py`` has produced
``data/fintabnet/records.json`` + ``ground_truth.json`` + images.

Goal: confirm the FinTabNet loader produces GT in the shape the scorer
expects, and that real-2B cell-F1 is in the same ballpark as
OmniDocBench at the same budget (≥ 0.05 on at least one page).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from adaptive_inference.config.budgets import Budget
from adaptive_inference.config.prompts import load_prompt_template
from adaptive_inference.dataset.records import load_page_records
from adaptive_inference.inference.internvl2 import InternVL2Adapter
from adaptive_inference.scoring.cell_f1 import score_page


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", type=Path, default=Path("data/fintabnet/records.json"),
    )
    parser.add_argument(
        "--ground-truth", type=Path,
        default=Path("data/fintabnet/ground_truth.json"),
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data"),
        help="Root that PageRecord.image_path is relative to.",
    )
    parser.add_argument(
        "--prompt", type=Path,
        default=Path("configs/prompts/table_parse_v1.yaml"),
    )
    parser.add_argument(
        "--model-id", type=str, default="OpenGVLab/InternVL2-2B",
    )
    parser.add_argument(
        "--model-name", type=str, default="internvl2-2b",
    )
    parser.add_argument("--max-tiles", type=int, default=10)
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    records = load_page_records(args.records)
    gt = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    if len(records) < args.n:
        print(f"only {len(records)} records available; using all of them")
    pages = records[: args.n]

    prompt = load_prompt_template(args.prompt)
    budget = Budget(name="smoke_b10", max_tiles=args.max_tiles)

    print(f"loading {args.model_id} ...")
    adapter = InternVL2Adapter(
        model_name=args.model_name,
        model_id=args.model_id,
    )
    print(f"device={adapter.device} dtype={adapter.dtype}")

    pass_count = 0
    for rec in pages:
        img_path = args.data_root / rec.image_path
        image = Image.open(img_path).convert("RGB")
        result = adapter.run(rec.page_id, image, budget, prompt)
        gold = gt[rec.page_id]
        score = score_page(result.raw_text, gold)
        ok = score.cell_f1 >= 0.05
        pass_count += int(ok)
        print(
            f"  {rec.page_id:30s} "
            f"cell_f1={score.cell_f1:.3f} "
            f"text_sim={score.text_similarity:.3f} "
            f"tokens={result.output_token_count:4d} "
            f"runtime={result.runtime_ms/1000:5.1f}s "
            f"parse_err={'-' if not score.pred_parse_error else score.pred_parse_error[:40]}"
        )

    print(f"\nG4 gate: {pass_count}/{len(pages)} pages cleared cell_f1 >= 0.05")
    return 0 if pass_count >= 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
