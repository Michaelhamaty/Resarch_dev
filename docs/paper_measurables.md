
## 2026-05-17 — Phase 2 + Phase 4 on real OmniDocBench (20-page calib)
After pinning `transformers==4.44.2`, dropping notebook patches B–E, and
adding the markdown-pipe-table → HTML scorer normalizer.

| run | macro_cell_f1 | macro_text_sim | parse_errors / 20 |
|---|---:|---:|---:|
| Phase 2 single-pass low (4 tiles)         | 0.0298 | 0.1298 | 11 |
| Phase 4 adaptive (4 → 12 on reparse)      | 0.0644 | 0.2939 |  4 |

Adaptive lift: 2.16× cell-F1, 7/11 parse-error pages rescued.
Not yet matched-cost (adaptive ~2.65× baseline tile budget) — Phase 5/6 next.

## 2026-05-17 — Phase 6 + Phase 7 on real OmniDocBench (20-page calib, in-sample)

Real five-system matched-cost run on the 20-page OmniDocBench English-table
calibration split. **No held-out split exists yet — all numbers are in-sample
for Phase 5's budget pick.** 8B row is stub output.

| system | macro_cell_f1 | macro_text_sim | parse_errors / 20 | cost (tiles/page) |
|---|---:|---:|---:|---:|
| adaptive_2b           | 0.1272 | 0.2836 |  6 |  9.5 |
| fixed_2b_low          | 0.0298 | 0.1298 | 11 |  4.0 |
| fixed_2b_matched      | 0.1861 | 0.3329 |  9 | 10.0 |
| random_2b_seed0       | 0.0461 | 0.1635 | 10 |  7.0 |
| random_2b_seed1       | 0.1038 | 0.2354 | 10 | 11.5 |
| random_2b_seed2       | 0.1609 | 0.2780 | 11 |  8.5 |
| fixed_8b_matched (stub) | 0.0000 | 0.0764 |  0 | 10.0 |

Matched-cost finding: adaptive_2b underperforms fixed_2b_matched by 0.0589
absolute cell-F1 at near-matched cost (9.5 vs 10 tiles/page). The deterministic
verifier (NO_TABLE_FOUND only, 11/20 pages) rescues 5 parse-error pages but
misses the 9 verifier-passing pages that would still gain quality at 10 tiles.

Phase 7 audit: 9 ok, 2 warn (splits_identical MVP shortcut, accuracy_status
stub-adapter gate), 0 fail. Artifacts under
`outputs/analysis/phase6_omnidocbench_v1/`.

## 2026-05-29 — Stage 6 re-validation under the surgical loop-stop (50-pp calib splits)

The InternVL2 adapter now carries the exact-periodicity runaway loop-stop
(commit `6e2f89e`): it only **truncates** degenerate exact-repeat token tails
and never alters token choices. Re-ran the *frozen* budget points on each
50-page calibration split to confirm the loop-stop is cost-neutral (so the
frozen Stage-6 budgets remain valid). Single-point grids written to separate
`*_validate` artifacts; committed budgets not overwritten by the run.

| dataset | budgets (B_low/B_high/B_fix) | reparse | adaptive cost_tiles (committed → re-measured) | fixed within_tol |
|---|---|---:|---:|:--:|
| OmniDocBench | 2 / 12 / 11 | 0.72 (36/50) | 10.64 → **10.64** (identical) | true / true |
| FinTabNet    | 6 / 16 / 10 | 0.26→**0.28** (13→14/50) | 10.16 → **10.48** | true / true |

**OmniDoc:** byte-for-byte cost-neutral — same 36 pages reparse, cost unchanged.
**FinTab:** one extra borderline page reparses under the loop-stop (deterministic,
not noise), nudging adaptive cost 10.16 → 10.48 (3.1%, within the 0.10 matched
tolerance). Budgets (tile counts) are **unchanged** — single-point grid, no
re-selection possible.

**Action — re-froze FinTab `measured_cost_tiles` 10.16 → 10.48** in
`configs/calibration/frozen_budgets_v2_fintabnet.json` (also the two fixed
`target_cost_tiles`). Rationale: Stage 7's random-escalation probability is
derived as `p = (measured_cost_tiles − B_low) / B_high`
(`experiment/frozen_inputs.py:derive_calibration_reparse_rate`). Using the
loop-stop-measured 10.48 sets random `p = 0.28`, so adaptive / random /
fixed_matched all sit at 10.48 tiles/page — preserving exact matched-cost
fairness under the adapter Stage 7 actually runs. OmniDoc needed no change.

**Loop-stop behavioural note (accepted tradeoff):** the stopper reliably catches
*low-budget* single-token loops (B_low pages drop from the ~64s/2048-token cap to
seconds) but does **not** catch *matched/high-budget* runaways, whose tails are
not exactly periodic within the 100-token window — fixed_2b@matched still caps
~30% of pages at 2048 tok (~64s), and 8B@matched runaways cost ~157s/page. Net:
calibration is now affordable and the low-budget worst case is bounded, but the
full Stage 7 sweep remains in the ~15–19 GPU-hr envelope (8B dominates).
Validation pace (mean total/page): OmniDoc adaptive 57.8s, FinTab adaptive 22.1s.
