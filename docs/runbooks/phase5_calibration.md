# Phase 5 runbook — budget calibration

Phase 5 freezes the four budgets Phase 6 needs:

- `B_low` — adaptive 2B first pass
- `B_high` — adaptive 2B reparse
- `B_fix_2B` — fixed-cost 2B matched-cost baseline
- `B_fix_8B` — fixed-cost 8B matched-cost baseline

Contract 6 ("calibration frozen before held-out evaluation") is
enforced by construction: this phase reads the **calibration split
only**, writes a frozen JSON artifact, and does not touch the
held-out split.

## What it produces

```
configs/calibration/
└── frozen_budgets.json       # the Phase-6 contract (committed)
outputs/calibration/
├── sweep_summaries.jsonl     # one line per sweep point (adaptive + fixed)
└── sweep/
    ├── adaptive/low_{L}_high_{H}/   # run_adaptive output dir per pair
    │   ├── first_pass/ reparse/ final/
    │   └── run.log.jsonl
    └── fixed_{model}/tiles_{T}/     # run_single_pass output dir per T
        ├── raw/ pages/
        └── run.log.jsonl
```

`frozen_budgets.json` carries the four budgets, the selection results,
and pinning metadata (calibration-split SHA256, cost unit, tolerance,
timestamp, calibration config path).

## Cost unit and selection rules

Cost is expressed in `max_tiles` (the real InternVL2 compute knob):

- **Adaptive cost per page** = `B_low + reparse_triggered * B_high`
  (both passes are paid when reparse fires).
- **Fixed cost per page** = `B.max_tiles`.

Selection:

1. `select_adaptive_pair` picks the `(low, high)` whose mean cost is
   closest to `target_adaptive_cost_tiles`. Tie-break: smallest `low`,
   then smallest `high`. Pairs with `high < low` are skipped.
2. `select_matched_fixed` picks the fixed candidate with relative
   deviation `≤ matched_cost_tolerance` from the adaptive pair's
   measured cost. Tie-break: smallest `|cost - target|`, then smallest
   `max_tiles`. When no candidate is inside the band, falls back to
   absolute-closest and records `within_tolerance = false` in the
   artifact.

No quality metric is consulted at this phase — Phase 6/7 owns TEDS
and edit-distance. Phase 5 is pure cost calibration.

## Stub adapter note

The repo ships stub adapters (`adapter_kind: stub`) that ignore
`max_tiles` and emit deterministic output. Under stubs, `reparse_rate`
is 0 for every adaptive pair, so adaptive cost collapses to `B_low`
and the numeric picks are placeholders until a real InternVL2 adapter
lands. The **artifact format, selection logic, and tests are real**
and will survive that swap without changes.

## How to run

```bash
uv sync --extra dev
uv run python scripts/fixtures/generate_placeholder_images.py  # once
uv run python scripts/calibration/run_calibration.py \
    --config configs/calibration/phase5.yaml
```

Expected output (on stub adapters with the shipped grid):

```
run_id                : phase5_calibration_v1
calibration split     : data/splits/calibration_split.json
adaptive pair         : B_low(max_tiles=6), B_high(max_tiles=8) | measured=6.000, target=6.000
B_fix_2B              : max_tiles=6 | measured=6.000, within_tolerance=True
B_fix_8B              : max_tiles=6 | measured=6.000, within_tolerance=True
frozen_budgets_path   : configs/calibration/frozen_budgets.json
sweep_summaries_path  : outputs/calibration/sweep_summaries.jsonl
```

## How to change the sweep

Edit `configs/calibration/phase5.yaml`:

- `candidates.low_max_tiles`, `candidates.high_max_tiles` — adaptive
  grid. Cartesian product; pairs with `high < low` are auto-skipped.
- `candidates.fixed_2b_max_tiles`, `candidates.fixed_8b_max_tiles` —
  per-model fixed grids.
- `target_adaptive_cost_tiles` — the cost the adaptive sweep aims for.
- `matched_cost_tolerance` — relative band for fixed-cost matching
  (e.g. `0.10` = ±10%).

No code changes are needed to re-sweep.

## Module map

```
src/adaptive_inference/calibration/
├── cost.py          # adaptive_cost_tiles / fixed_cost_tiles
├── summary.py       # SweepPointSummary + summarize_{adaptive,single_pass}_log
├── sweep.py         # sweep_adaptive / sweep_fixed (call run_adaptive / run_single_pass)
├── select.py        # select_adaptive_pair / select_matched_fixed
├── artifact.py      # FrozenBudgets + write_/load_frozen_budgets
└── config.py        # CalibrationConfig loader
```

`scripts/calibration/run_calibration.py` glues them together. The
calibration layer never modifies the Phase 2/4 runners, verifier,
adapters, or existing configs.
