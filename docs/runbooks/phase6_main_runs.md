# Phase 6 — Main Experimental Runs

## What this phase does

Execute the five required systems on the **held-out evaluation split**,
using the budgets frozen by Phase 5, and write complete per-page
artifacts plus a top-level run manifest that Phase 7 consumes.

Phase 6 does not retune budgets, does not recalibrate, does not alter
the verifier or the adaptive pipeline. It is a thin orchestration layer
over the existing single-pass and adaptive runners.

## Systems executed

| system_id            | runner           | budgets          | escalation policy   |
|----------------------|------------------|------------------|---------------------|
| `adaptive_2b`        | adaptive         | `B_low`→`B_high` | verifier-based      |
| `fixed_2b_low`       | single_pass      | `B_low`          | n/a                 |
| `fixed_2b_matched`   | single_pass      | `B_fix_2B`       | n/a                 |
| `random_2b_seed{0,1,2}` | adaptive_random | `B_low`→`B_high` | seeded random at p  |
| `fixed_8b_matched`   | single_pass      | `B_fix_8B`       | n/a                 |

`p` is the calibration-measured reparse rate, derived from the Phase 5
artifact as `(measured_cost_tiles − low_max_tiles) / high_max_tiles`
(additive cost model). If the Phase 5 adaptive system never escalated
on calibration, `p == 0.0` and the manifest marks it as `degenerate`.

## Prerequisites

1. Phase 5 completed. File must exist:
   `configs/calibration/frozen_budgets.json`.
2. Held-out split manifest present:
   `data/splits/held_out_eval_split.json`.
3. Page records and images reachable from
   `data/fixtures/sample_pages.json` + `data/` (the existing fixture setup).

## How to run

Execute every system:

```bash
uv run python scripts/main_runs/run_phase6.py \
  --config configs/experiment/phase6.yaml
```

Without `--allow-stubbed-8b`, the `fixed_8b_matched` system is
**skipped** when the frozen `B_fix_8B` points at a stub adapter. The
other four systems still run. The manifest records
`status="skipped_stub_8b"` and the reason.

To include the stubbed 8B (useful for pipeline validation; does NOT
produce a real 8B number):

```bash
uv run python scripts/main_runs/run_phase6.py \
  --config configs/experiment/phase6.yaml \
  --allow-stubbed-8b
```

To run only a subset (e.g. re-run the random seeds after a fix):

```bash
uv run python scripts/main_runs/run_phase6.py \
  --config configs/experiment/phase6.yaml \
  --systems random_2b
```

Specific seed:

```bash
uv run python scripts/main_runs/run_phase6.py \
  --config configs/experiment/phase6.yaml \
  --systems random_2b_seed1
```

## Outputs

```
outputs/runs/phase6/
├── manifest.json                  ← Phase 7 entry point
├── adaptive_2b/
│   ├── first_pass/{raw,pages}/
│   ├── reparse/{raw,pages}/       (only pages that escalated)
│   ├── final/{raw,pages}/
│   └── run.log.jsonl
├── fixed_2b_low/
│   ├── raw/*.md
│   ├── pages/*.json
│   └── run.log.jsonl
├── fixed_2b_matched/              (same layout as fixed_2b_low)
├── random_2b_seed0/               (adaptive layout)
├── random_2b_seed1/
├── random_2b_seed2/
└── fixed_8b_matched/              (single-pass layout, or missing if skipped)
```

The manifest is the single source of truth for Phase 7. Every system
entry records its output dir, budget identities, adapter kind, and
status — failures and skips are explicit, never silent.

## Exit codes

- `0`: every attempted system completed (skipped systems are not failures).
- `1`: at least one system raised and was recorded as `status="failed"`.
  Inspect the `error` field in `manifest.json` for the traceback.

## What Phase 6 does NOT do

- Re-calibrate budgets (those are frozen).
- Run TEDS or any accuracy scoring (that is Phase 7).
- Implement a real InternVL2 adapter. When one lands, flip
  `configs/models/internvl2.yaml` from `adapter_kind: stub` to the real
  kind and re-freeze Phase 5; Phase 6 picks up the new adapter
  automatically.
- Add cross-system resume state. Per-page idempotency is inherited from
  the existing runners (`reset_adaptive_log` / sidecar overwrite).
