"""GPU validation for the surgical runaway loop-stop (InternVL2Adapter).

Background: InternVL2-2B greedy-decodes into degenerate exact-repeat loops
on many hard pages, running to the max_new_tokens cap (~64s on an L4). This
is genuine model behavior, not a config bug, and it cannot be removed by
repetition penalties (unreliable + corrupt real tables) or a length cap
(real outputs overlap the runaways in length). The adapter instead halts
generation only once the trailing window is *exactly periodic* — a stopping
criterion that never alters token choices.

This script proves two things by running the SAME pages with the loop-stop
OFF (loop_stop_window=0) then ON (default), at two budgets:

  1. SPEED: on pages that run away at the LOW budget, ON truncates the loop
     early (far fewer tokens, far less time).
  2. SAFETY: on pages that are HEALTHY at the matched budget, ON produces
     output BYTE-IDENTICAL to OFF (the stopper never fires) — which is what
     keeps the frozen Stage-6 calibration valid.

    uv run python scripts/scaleup/ab_repetition_check.py --dataset omnidocbench
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from adaptive_inference.config.budgets import Budget  # noqa: E402
from adaptive_inference.config.prompts import load_prompt_template  # noqa: E402
from adaptive_inference.inference.internvl2 import InternVL2Adapter  # noqa: E402
from adaptive_inference.runner.pages import load_pages_for_manifest  # noqa: E402

MODEL_ID = "OpenGVLab/InternVL2-2B"
MAX_NEW_TOKENS = 2048
PROMPT_PATH = "configs/prompts/table_parse_v2.yaml"

DATASETS = {
    "omnidocbench": {
        "split": "data/splits/scaleup_v2/omnidocbench/held_out.json",
        "records": "data/omnidocbench/records.json",
        "low_tiles": 2,
        "matched_tiles": 11,
        # Mix of confirmed runaways + a confirmed-healthy page.
        "pages": [
            "PPT_8076_MEYER_Chapter_2_-_Language_Change_page_021",
            "PPT_Catalysis_ppt_page_016",
            "book_en______893__Pyomo_Optimization_Modeling_in_Python__2012_____page_188",
            "color_textbook_zhonggaokao____13_________4-5___________________________5B_____page_036",
            "docstructbench_enbook-zlib-o_O-17761417_pdf_894",
        ],
    },
    "fintabnet": {
        "split": "data/splits/scaleup_v2/fintabnet/held_out.json",
        "records": "data/fintabnet/records.json",
        "low_tiles": 6,
        "matched_tiles": 10,
        "pages": [],
    },
}


def _collect(adapter, pages, budgets, prompt):
    out = {}
    for bname, budget in budgets.items():
        for p in pages:
            r = adapter.run(p.record.page_id, p.image, budget, prompt)
            out[(p.record.page_id, bname)] = (
                r.output_token_count,
                r.runtime_ms,
                r.raw_text,
            )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="omnidocbench", choices=sorted(DATASETS))
    args = parser.parse_args(argv)

    ds = DATASETS[args.dataset]
    prompt = load_prompt_template(PROMPT_PATH)
    all_pages = load_pages_for_manifest(ds["split"], ds["records"], "data")
    by_id = {p.record.page_id: p for p in all_pages}
    pages = [by_id[p] for p in ds["pages"] if p in by_id] or all_pages[:2]
    budgets = {
        "LOW": Budget(name="low", max_tiles=ds["low_tiles"]),
        "MATCHED": Budget(name="matched", max_tiles=ds["matched_tiles"]),
    }
    print(f"[probe] dataset={args.dataset} pages={[p.record.page_id for p in pages]}")

    import torch

    print("[probe] === OFF (loop_stop disabled) ===")
    off_adapter = InternVL2Adapter(
        model_name="internvl2-2b", model_id=MODEL_ID, loop_stop_window=0
    )
    off = _collect(off_adapter, pages, budgets, prompt)
    del off_adapter
    gc.collect()
    torch.cuda.empty_cache()

    print("[probe] === ON (default loop_stop) ===")
    on_adapter = InternVL2Adapter(model_name="internvl2-2b", model_id=MODEL_ID)
    on = _collect(on_adapter, pages, budgets, prompt)

    print("[probe] === RESULTS (tok/ms; identical = healthy output unchanged) ===")
    for key in off:
        pid, bname = key
        otok, oms, otext = off[key]
        ntok, nms, ntext = on[key]
        identical = otext == ntext
        capped = otok >= MAX_NEW_TOKENS - 5
        speedup = (oms / nms) if nms else 1.0
        tag = "IDENTICAL" if identical else f"CHANGED(tr {otext.count('<tr>')}->{ntext.count('<tr>')})"
        runaway = " [OFF-RUNAWAY]" if capped else ""
        print(
            f"[{bname:7}] {pid[:30]:30} OFF tok={otok:>4} ms={oms:>6.0f} | "
            f"ON tok={ntok:>4} ms={nms:>6.0f} | x{speedup:>4.1f} | {tag}{runaway}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
